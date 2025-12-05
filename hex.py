from sgptools.methods import *

class HexCoverage(Method):
    """
    Hexagonal lattice coverage based on kernel hyperparameters.

    This method constructs a deterministic hexagonal tiling over a rectangular
    environment such that the GP posterior variance at every point in the
    environment is bounded by a user-specified threshold (under the same
    sufficient condition used in the minimal HexCover implementation).

    Notes
    -----
    - Only supports 2D spatial domains (first two coordinates).
    - Multi-robot settings are not supported (`num_robots` must be 1).
    - The total number of points is determined by the tiling; it may be
      different from `num_sensing`. As with `GreedyCoverage`, returning fewer
      than `num_sensing` points is allowed.
    """

    def __init__(self,
                 num_sensing: int,
                 X_objective: np.ndarray,
                 kernel: gpflow.kernels.Kernel,
                 noise_variance: float,
                 transform: Optional[Transform] = None,
                 num_robots: int = 1,
                 X_candidates: Optional[np.ndarray] = None,
                 num_dim: Optional[int] = None,
                 height: Optional[float] = None,
                 width: Optional[float] = None,
                 **kwargs: Any):
        """
        Initialize a HexCoverage method.

        Parameters
        ----------
        num_sensing : int
            Maximum number of sensing locations (not strictly enforced; the
            tiling determines the actual number of points).
        X_objective : ndarray, shape (n, d)
            Environment points. Used only to infer the bounding rectangle
            (min/max in the first two dimensions) when `height`/`width` are
            not provided.
        kernel : gpflow.kernels.Kernel
            GP kernel (assumed to have a `variance` and `lengthscales`
            attribute, e.g., SquaredExponential).
        noise_variance : float
            Observation noise variance.
        transform : Transform or None
            Reserved for compatibility with other methods.
        num_robots : int
            Must be 1. Multi-robot tilings are not supported.
        X_candidates : ndarray or None
            Ignored. Present for API compatibility with other methods.
        num_dim : int or None
            Dimensionality of points. Defaults to `X_objective.shape[-1]`.
        height : float or None
            Environment height in the y-direction. If None, inferred from
            `X_objective` as `y_max - y_min`.
        width : float or None
            Environment width in the x-direction. If None, inferred from
            `X_objective` as `x_max - x_min`.
        kwargs : dict
            Ignored. Accepted for forward compatibility.
        """
        super().__init__(num_sensing, X_objective, kernel, noise_variance,
                         transform, num_robots, X_candidates, num_dim)

        assert num_robots == 1, "HexCoverage only supports num_robots = 1."

        self.kernel = kernel
        self.noise_variance = float(noise_variance)

        # Store environment points for dtype and potential debugging
        self.X_objective = np.asarray(X_objective)

        if self.X_objective.ndim != 2 or self.X_objective.shape[1] < 2:
            raise ValueError(
                "HexCoverage requires X_objective with at least 2 spatial dimensions."
            )

        # Bounding box of the environment in (x, y) from X_objective
        mins = self.X_objective[:, :2].min(axis=0)
        maxs = self.X_objective[:, :2].max(axis=0)
        default_extent = maxs - mins

        self.origin = mins  # shift from [0, W]x[0, H] to actual coords
        self.width = float(width) if width is not None else float(default_extent[0])
        self.height = float(height) if height is not None else float(default_extent[1])

        # Extract scalar lengthscale and prior variance
        self.lengthscale = self._extract_kernel_scalar(
            self.kernel, "lengthscales", "lengthscale"
        )
        self.prior_variance = self._extract_kernel_scalar(
            self.kernel, "variance", "prior variance"
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_kernel_scalar(
        kernel: gpflow.kernels.Kernel,
        attr: str,
        human_name: str
    ) -> float:
        """
        Extract a scalar float from a gpflow kernel attribute (e.g. lengthscales
        or variance). If the underlying value is vector-valued, returns the mean.

        Parameters
        ----------
        kernel : gpflow.kernels.Kernel
            Kernel object with the desired attribute.
        attr : str
            Name of the attribute on the kernel (e.g. 'lengthscales', 'variance').
        human_name : str
            Human-readable name for error messages.

        Returns
        -------
        float
            Scalar value for the requested kernel attribute.
        """
        if not hasattr(kernel, attr):
            raise ValueError(
                f"HexCoverage requires a kernel with a '{attr}' attribute "
                f"(for {human_name})."
            )

        value = getattr(kernel, attr)
        try:
            value = value.numpy()
        except AttributeError:
            pass

        value = np.asarray(value, dtype=float)
        return float(value.mean())

    def _compute_rmin(self, var_threshold: Optional[float] = None) -> float:
        """
        Compute the sufficient radius r_min for the hexagonal tiling, following
        the same condition as in the minimal HexCover implementation:

            r_min = L * sqrt(-log((σ₀² - Δ)(σ₀² + σ²) / σ₀⁴))

        where L is the kernel lengthscale, σ₀² is the prior variance, σ² is the
        noise variance, and Δ is the allowed posterior variance threshold.
        """
        if var_threshold is None:
            var_threshold = self.var_threshold

        sigma0_sq = self.prior_variance
        sigma_sq = float(self.noise_variance)
        delta = float(var_threshold)
        term = (sigma0_sq - delta) * (sigma0_sq + sigma_sq) / (sigma0_sq ** 2)

        if term <= 0.0 or term >= 1.0:
            raise ValueError(
                f"Invalid term inside log when computing r_min: {term}. "
                "Check kernel hyperparameters and var_threshold."
            )

        return self.lengthscale * np.sqrt(-np.log(term))

    @staticmethod
    def _hexagonal_tiling(height: float,
                          width: float,
                          radius: float,
                          fill_edge: bool = True) -> np.ndarray:
        """
        Hexagonal tiling helper, ported from minimal.py's `hexagonal_tiling`.

        Parameters
        ----------
        height : float
            Height of environment in y-direction.
        width : float
            Width of environment in x-direction.
        radius : float
            Hexagon circumradius r_min.
        fill_edge : bool
            If True, adds additional centers near the environment boundary.
        """
        hs = 3.0 * radius
        vs = np.sqrt(3.0) * radius

        # first set of centers
        nc = int(np.floor(width / hs) + 1)
        nr = int(np.floor(height / vs) + 1)
        x = list(np.linspace(0.0, (nc - 1) * hs, nc))
        y = list(np.linspace(0.0, (nr - 1) * vs, nr))

        if fill_edge:
            if (nc - 1) * hs + radius < width:
                x.append(width)
            if (nr - 1) * vs + radius < height:
                y.append(height)

        X, Y = np.meshgrid(x, y)
        first_centers = np.stack([X.ravel(), Y.ravel()], axis=-1)

        # second set of centers (offset grid)
        nc = int(np.floor((width / hs) + 0.5))
        nr = int(np.floor((height / vs) + 0.5))
        x = list(np.linspace(hs / 2.0, (nc - 1) * hs + hs / 2.0, nc))
        y = list(np.linspace(vs / 2.0, (nr - 1) * vs + vs / 2.0, nr))

        if fill_edge:
            if (nc - 1) * hs + hs / 2.0 + radius < width:
                x.append(width)
            if (nr - 1) * vs + vs / 2.0 + radius < height:
                y.append(height)

        X, Y = np.meshgrid(x, y)
        second_centers = np.stack([X.ravel(), Y.ravel()], axis=-1)

        return np.concatenate([first_centers, second_centers], axis=0)

    # ------------------------------------------------------------------
    # Method API
    # ------------------------------------------------------------------
    def update(self,
               kernel: gpflow.kernels.Kernel,
               noise_variance: float) -> None:
        """
        Update kernel and noise variance hyperparameters.
        """
        self.kernel = kernel
        self.noise_variance = float(noise_variance)
        
        self.lengthscale = self._extract_kernel_scalar(
            self.kernel, "lengthscales", "lengthscale"
        )
        self.prior_variance = self._extract_kernel_scalar(
            self.kernel, "variance", "prior variance"
        )

    def get_hyperparameters(self) -> Tuple[gpflow.kernels.Kernel, float]:
        """
        Return current kernel and noise variance as (kernel, noise_variance).
        """
        return deepcopy(self.kernel), float(self.noise_variance)

    def optimize(self,
                 var_threshold: Optional[float] = None,
                 return_fovs: bool = False,
                 **kwargs: Any) -> np.ndarray:
        """
        Construct the hexagonal coverage pattern.

        Parameters
        ----------
        var_threshold : float or None
            Target posterior variance threshold Δ. If None, defaults to
            `0.2 * variance` (following the minimal implementation where
            `delta = 0.2 * sigma0**2`).
        return_fovs : bool, optional
            If True, also returns a list of polygonal fields of view (FoVs)
            corresponding to the convex hull of the covered objective points
            for each selected candidate. Default is False.
            
        Returns
        -------
        X_sol : ndarray of shape (1, k, d)
            Selected sensing locations. `k` is determined by the tiling, and
            may differ from `num_sensing`. The last dimension `d` matches
            `self.num_dim`; only the first two coordinates are used for the
            spatial layout, the remaining coordinates are zero.

        If `return_fovs` is True, the return value is:

        -------
        X_sol, fovs : (ndarray, list of shapely.geometry.Polygon)
            `X_sol` as above, together with a list of buffered convex-hull FoVs
            computed from the covered environment points for each selected
            candidate.
        """
        # Posterior variance threshold Δ
        if var_threshold is None:
            self.var_threshold = 0.2 * self.prior_variance
        else:
            self.var_threshold = float(var_threshold)

        if self.var_threshold >= self.prior_variance:
            raise ValueError(
                f"var_threshold must be smaller than the kernel variance: {self.prior_variance:.2f}."
            )

        # Compute r_min for the current kernel / noise / threshold
        rmin = self._compute_rmin(var_threshold)

        # Tiling in local [0, width] x [0, height] coordinates
        centers_2d = self._hexagonal_tiling(self.height, self.width, rmin)

        # Shift to actual environment coordinates using the inferred origin
        centers_2d = centers_2d + self.origin[None, :]

        # Embed in full d-dimensional space
        dtype = self.X_objective.dtype
        k = centers_2d.shape[0]
        X_sol = np.zeros((k, self.num_dim), dtype=dtype)
        X_sol[:, :2] = centers_2d

        X_sol, _ = run_tsp(X_sol, **kwargs)    
        X_sol = np.array(X_sol).reshape(self.num_robots, -1, self.num_dim)

        if return_fovs:
            return X_sol, self._get_fovs(X_sol, rmin)
        else:
            return X_sol

    def _get_fovs(self, X_sol, radius):
        """
        Compute polygonal fields of view (FoVs) from coverage masks.

        For each selected candidate, this method takes the subset of objective
        points it covers, forms their convex hull, and then applies a
        morphological buffer. The result is a list of polygons that roughly
        characterize the spatial footprint of each sensing location.

        Parameters
        ----------
        coverages : 2D ndarray of bool, shape (k, n)
            Coverage masks for the selected candidates, where `k` is the number
            of selected locations and `n` is the number of environment points.
            Each row corresponds to one candidate and indicates which objective
            points are covered.
        buffer : float, optional
            Buffer radius passed to `Polygon.buffer`. Controls how much
            the convex hull is expanded. Default is 0.5.

        Returns
        -------
        fovs : list of shapely.geometry.Polygon
            List of buffered convex-hull polygons, one per candidate for which
            at least four objective points are covered. Candidates covering
            fewer than four points are skipped.
        """
        fovs = []
        for pt in X_sol[0]:
            fov = geometry.Point(pt[0], pt[1])
            fov = fov.buffer(radius)
            fovs.append(fov)
        return fovs
    
METHODS['HexCoverage'] = HexCoverage