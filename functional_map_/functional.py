import copy
import time

import numpy as np
import functional_map_.geometry as geom
from scipy.optimize import fmin_l_bfgs_b
import scipy.sparse as sparse
import functional_map_
import robust_laplacian
from functional_map_ import spectral as spectral, optimize as opt_func, refine, signatures as sg


def l2_sqnorm(A, func):
    if len(func.shape) == 1:
        func = func[:, None]
        return np.einsum('np,np->p', func, A @ func).flatten().item()
    return np.einsum('np,np->p', func, A @ func).flatten()


class FunctionalMapping:
    """
    A class to compute functional maps between two meshes

    Attributes
    ----------------------
    mesh1  : TriMesh - first mesh
    mesh2  : TriMesh - second mesh

    descr1 : (n1,p) signatures on the first mesh
    descr2 : (n2,p) signatures on the second mesh
    D_a    : (k1,k1) area-based shape differnence operator
    D_c    : (k1,k1) conformal-based shape differnence operator

    Properties
    ----------------------
    FM_type : 'classic' | 'icp' | 'zoomout' which FM is currently used
    k1      : dimension of the first eigenspace (varies depending on the type of FM)
    k2      : dimension of the seconde eigenspace (varies depending on the type of FM)
    FM      : (k2,k1) current FM
    p2p     : (n2,) point to point map associated to the current functional map
    """

    def __init__(self, mesh1, mesh2):

        self.mesh1 = copy.deepcopy(mesh1)
        self.mesh2 = copy.deepcopy(mesh2)

        self.mesh1.compute_triangle_normals()
        self.mesh2.compute_triangle_normals()

        self.eigenvalues1, self.eigenvectors1 = None, None
        self.eigenvalues2, self.eigenvectors2 = None, None

        self.A1 = None
        self.A2 = None

        self.normals1 = None
        self.normals2 = None

        # DESCRIPTORS
        self.descr1 = None
        self.descr2 = None

        # FUNCTIONAL MAP
        self._FM_type = 'classic'
        self.FM_base = None
        self.FM_icp = None
        self.FM_zo = None

        # AREA AND CONFORMAL SHAPE DIFFERENCE OPERATORS
        self.SD_a = None
        self.SD_c = None

        self._k1, self._k2 = None, None
        self.max_dim = 0

    # DIMENSION PROPERTIES
    @property
    def k1(self):
        """"
        Return the input dimension of the functional map
        """
        if self._k1 is None and not self.preprocessed and not self.fitted:
            raise ValueError('No information known about dimensions')
        if self.fitted:
            return self.FM.shape[1]
        else:
            return self._k1

    @k1.setter
    def k1(self, k1):
        self._k1 = k1

    @property
    def k2(self):
        if self._k2 is None and not self.preprocessed and not self.fitted:
            raise ValueError('No information known about dimensions')
        if self.fitted:
            return self.FM.shape[0]
        else:
            return self._k2

    @k2.setter
    def k2(self, k2):
        self._k2 = k2

    # FUNCTIONAL MAP SWITCHER (REFINED OR NOT)
    @property
    def FM_type(self):
        return self._FM_type

    @FM_type.setter
    def FM_type(self, FM_type):
        if FM_type.lower() not in ['classic', 'icp', 'zoomout']:
            raise ValueError(f'FM_type can only be set to "classic", "icp" or "zoomout", not {FM_type}')
        self._FM_type = FM_type

    def change_FM_type(self, FM_type):
        self.FM_type = FM_type

    @property
    def FM(self):
        """
        Returns the current functional map depending on the value of FM_type

        Output
        ----------------
        FM : (k2,k1) current FM
        """
        if self.FM_type.lower() == 'classic':
            return self.FM_base
        elif self.FM_type.lower() == 'icp':
            return self.FM_icp
        elif self.FM_type.lower() == 'zoomout':
            return self.FM_zo

    @FM.setter
    def FM(self, FM):
        self.FM_base = FM

    # BOOLEAN PROPERTIES
    @property
    def preprocessed(self):
        """
        check if enough information is provided to fit the model
        """
        test_descr = (self.descr1 is not None) and (self.descr2 is not None)
        test_evals = (self.eigenvalues1 is not None) and (self.eigenvalues2 is not None)
        test_evects = (self.eigenvectors1 is not None) and (self.eigenvectors2 is not None)
        return test_descr and test_evals and test_evects

    @property
    def fitted(self):
        return self.FM is not None

    @property
    def p2p(self):
        """
        Computes the current point to point map, going from the second mesh to the first one.

        Output
        --------------------
        p2p : (n2,) point to point map associated to the current functional map
        """
        if not self.fitted:
            raise ValueError('Model should be fit and fit to obtain p2p map')

        return spectral.FM_to_p2p(self.FM, self.eigenvectors1, self.eigenvectors2)

    def precise_map(self, precompute=True):
        """
        Returns a precise map between the two meshes using the map deblurring paper

        Paramaters
        -------------------
        precompute : Set to precompute values for faster computation but heavier
                     memory use.

        Output
        -------------------
        precise_map : (n2,n1) sparse - precise map
        """
        if not self.fitted:
            raise ValueError('Model should be fit and fit to obtain p2p map')

        return functional_map_.spectral.precise_map.precise_map(self.mesh1, self.mesh2, self.FM,
                                                                precompute_dmin=precompute)

    def preprocess(self, n_ev=(50, 50), n_descr=100, descr_type='WKS', landmarks=None, subsample_step=1, verbose=False):
        """
        Saves the information about the Laplacian mesh for opt

        Parameters
        -----------------------------
        n_ev           : (k1, k2) tuple - with the number of Laplacian eigenvalues to consider.
        n_descr        : int - number of signatures to consider
        descr_type     : str - "HKS" | "WKS"
        landmarks      : (p,1|2) array of indices of landmarks to match.
                         If (p,1) uses the same indices for both.
        subsample_step : int - step with which to subsample the signatures.
        """
        self.k1, self.k2 = n_ev

        use_lm = landmarks is not None and len(landmarks) > 0

        # Compute the Laplacian spectrum
        if verbose:
            print('\nComputing Laplacian spectrum')
        max_dim = min([self.k1 + 120,
                       np.asarray(self.mesh1.vertices).shape[0] - 10,
                       np.asarray(self.mesh2.vertices).shape[0] - 10])
        self.k1 = min(self.k1, max_dim)
        self.k2 = min(self.k2, max_dim)
        self.max_dim = max_dim
        self.preprocess_mesh(max(self.k1, max_dim), verbose=verbose)

        if verbose:
            print('\nComputing signatures')

        # Extract landmarks indices
        if use_lm:
            # if np.asarray(landmarks).squeeze().ndim == 1:
            #     if verbose:
            #         print('\tUsing same landmarks indices for both meshes')
            #     lm1 = np.asarray(landmarks).squeeze()
            #     lm2 = lm1
            # else:
                lm1, lm2 = landmarks[:, 0], landmarks[:, 1]

        # Compute signatures
        if descr_type == 'HKS':
            self.descr1 = sg.mesh_HKS(self.eigenvalues1, self.eigenvectors1, n_descr, k=self.k1)  # (N1, n_descr)
            self.descr2 = sg.mesh_HKS(self.eigenvalues2, self.eigenvectors2, n_descr, k=self.k2)  # (N2, n_descr)

            if use_lm:
                lm_descr1 = sg.mesh_HKS(self.eigenvalues1, self.eigenvectors1, n_descr, landmarks=lm1, k=self.k1)  # (N1, p*n_descr)
                lm_descr2 = sg.mesh_HKS(self.eigenvalues2, self.eigenvectors2, n_descr, landmarks=lm2, k=self.k2)  # (N2, p*n_descr)

                # lm_descr1 = np.zeros((self.descr1.shape[0], lm1.shape[0]))
                # lm_descr2 = np.zeros((self.descr1.shape[0], lm2.shape[0]))
                # lm_descr1[lm1] = np.eye(lm_descr1.shape[1])
                # lm_descr2[lm2] = np.eye(lm_descr2.shape[1])

                self.descr1 = np.hstack([self.descr1, lm_descr1])  # (N1, (p+1)*n_descr)
                self.descr2 = np.hstack([self.descr2, lm_descr2])  # (N2, (p+1)*n_descr)

        elif descr_type == 'WKS':
            self.descr1 = sg.mesh_WKS(self.eigenvalues1, self.eigenvectors1, n_descr, k=self.k1)  # (N1, n_descr)
            self.descr2 = sg.mesh_WKS(self.eigenvalues2, self.eigenvectors2, n_descr, k=self.k2)  # (N2, n_descr)

            if use_lm:
                lm_descr1 = sg.mesh_WKS(self.eigenvalues1, self.eigenvectors1, n_descr, landmarks=lm1, k=self.k1)  # (N1, p*n_descr)
                lm_descr2 = sg.mesh_WKS(self.eigenvalues2, self.eigenvectors2, n_descr, landmarks=lm2, k=self.k2)  # (N2, p*n_descr)

                # lm_descr1 = np.zeros((self.descr1.shape[0], lm1.shape[0]))
                # lm_descr2 = np.zeros((self.descr1.shape[0], lm2.shape[0]))
                # lm_descr1[lm1] = np.eye(lm_descr1.shape[1])
                # lm_descr2[lm2] = np.eye(lm_descr2.shape[1])

                self.descr1 = np.hstack([self.descr1, lm_descr1])  # (N1, (p+1)*n_descr)
                self.descr2 = np.hstack([self.descr2, lm_descr2])  # (N2, (p+1)*n_descr)

        elif descr_type == 'MIX':
            self.descr1 = np.hstack([ sg.mesh_HKS(self.eigenvalues1, self.eigenvectors1, n_descr, k=self.k1), # (N1, 2 * n_descr)
                                      sg.mesh_WKS(self.eigenvalues1, self.eigenvectors1, n_descr, k=self.k1)])
            self.descr2 = np.hstack([ sg.mesh_HKS(self.eigenvalues2, self.eigenvectors2, n_descr, k=self.k2), # (N1, 2 * n_descr)
                                      sg.mesh_WKS(self.eigenvalues2, self.eigenvectors2, n_descr, k=self.k2)]) # (N2, n_descr)
            if use_lm:
                lm_descr1 = np.hstack([ sg.mesh_HKS(self.eigenvalues1, self.eigenvectors1, n_descr, landmarks=lm1,
                                                    k=self.k1),
                                        sg.mesh_WKS(self.eigenvalues1, self.eigenvectors1, n_descr, landmarks=lm1,
                                                    k=self.k1)])
                lm_descr2 = np.hstack([ sg.mesh_HKS(self.eigenvalues2, self.eigenvectors2, n_descr, landmarks=lm2,
                                                    k=self.k2),
                                        sg.mesh_WKS(self.eigenvalues2, self.eigenvectors2, n_descr, landmarks=lm2,
                                                    k=self.k2)])

                self.descr1 = np.hstack([self.descr1, lm_descr1])  # (N1, 2 * (p+1)*n_descr)
                self.descr2 = np.hstack([self.descr2, lm_descr2])
        else:
            raise ValueError(f'Descriptor type "{descr_type}" not implemented')

        # Subsample signatures
        self.descr1 = self.descr1[:, np.arange(0, self.descr1.shape[1], subsample_step)]
        self.descr2 = self.descr2[:, np.arange(0, self.descr2.shape[1], subsample_step)]

        # Normalize signatures
        if verbose:
            print('\tNormalizing signatures')

        no1 = np.sqrt(l2_sqnorm(self.A1, self.descr1))  # (p,)
        no2 = np.sqrt(l2_sqnorm(self.A2, self.descr2))  # (p,)
        #
        self.descr1 /= no1[None, :]
        self.descr2 /= no2[None, :]

        if verbose:
            n_lmks = np.asarray(landmarks).shape[0] if use_lm else 0
            print(f'\n\t{self.descr1.shape[1]} out of {n_descr * (1 + n_lmks)} possible signatures kept')

        return self

    def preprocess_mesh(self, k, skip_normals=False, verbose=False):

        if (self.eigenvectors1 is not None) and (self.eigenvalues1 is not None)\
           and (len(self.eigenvalues1) >= k):
            self.eigenvectors1 = self.eigenvectors1[:,:k]
            self.eigenvalues1 = self.eigenvalues1[:k]
        else:
            v1 = np.asarray(self.mesh1.vertices)
            f1 = np.asarray(self.mesh1.triangles)
            L, M = robust_laplacian.mesh_laplacian(v1, f1)
            self.A1 = copy.deepcopy(M)
            self.eigenvalues1, self.eigenvectors1 = sparse.linalg.eigsh(L, k=k, M=M,
                                                            sigma=-0.01)

        if (self.eigenvectors2 is not None) and (self.eigenvalues2 is not None)\
           and (len(self.eigenvalues2) >= k):
            self.eigenvectors2 = self.eigenvectors2[:,:k]
            self.eigenvalues2 = self.eigenvalues2[:k]
        else:
            v2 = np.asarray(self.mesh2.vertices)
            f2 = np.asarray(self.mesh2.triangles)
            L, M = robust_laplacian.mesh_laplacian(v2, f2)
            self.A2 = copy.deepcopy(M)
            self.eigenvalues2, self.eigenvectors2 = sparse.linalg.eigsh(L, k=k, M=M,
                                                            sigma=-0.01)
        return

    def fit(self, descr_mu=1e-1, lap_mu=1e-3, descr_comm_mu=1, orient_mu=0, orient_reversing=False,
            optinit='zeros',
            verbose=False):
        """
        Solves the functional map optimization problem :

        min_C descr_mu * ||C@A - B||^2 + descr_comm_mu * (sum_i ||C@D_Ai - D_Bi@C||^2)
              + lap_mu * ||C@L1 - L2@C||^2 + orient_mu * (sum_i ||C@G_Ai - G_Bi@C||^2)

        with A and B signatures, D_Ai and D_Bi multiplicative operators extracted
        from the i-th signatures, L1 and L2 laplacian on each shape, G_Ai and G_Bi
        orientation preserving (or reversing) operators association to the i-th signatures.

        Parameters
        -------------------------------
        descr_mu         : scaling for the descriptor preservation term
        lap_mu           : scaling of the laplacian commutativity term
        descr_comm_mu    : scaling of the multiplicative operator commutativity
        orient_mu        : scaling of the orientation preservation term
                           (in addition to relative scaling with the other terms as in the original
                           code)
        orient_reversing : Whether to use the orientation reversing term instead of the orientation
                           preservation one
        optinit          : 'random' | 'identity' | 'zeros' initialization.
                           In any case, the first column of the functional map is computed by hand
                           and not modified during optimization
        """
        if optinit not in ['random', 'identity', 'zeros']:
            raise ValueError(f"optinit arg should be 'random', 'identity' or 'zeros', not {optinit}")

        if not self.preprocessed:
            self.preprocess()

        # Project the signatures on the LB basis
        descr1_red = self.project(self.descr1, mesh_ind=1)  # (n_ev1, n_descr)
        descr2_red = self.project(self.descr2, mesh_ind=2)  # (n_ev2, n_descr)

        # Compute multiplicative operators associated to each descriptor
        list_descr = []
        if descr_comm_mu > 0:
            if verbose:
                print('Computing commutativity operators')
            list_descr = self.compute_descr_op()  # (n_descr, ((k1,k1), (k2,k2)) )

        # Compute orientation operators associated to each descriptor
        orient_op = []
        if orient_mu > 0:
            if verbose:
                print('Computing orientation operators')
            orient_op = self.compute_orientation_op(reversing=orient_reversing)  # (n_descr,)

        # Compute the squared differences between eigenvalues for LB commutativity
        ev_sqdiff = np.square(
            self.eigenvalues1[None, :self.k1] - self.eigenvalues2[:self.k2, None])  # (n_ev2,n_ev1)
        ev_sqdiff /= np.linalg.norm(ev_sqdiff) ** 2

        # rescale orientation term
        if orient_mu > 0:
            args_native = (np.eye(self.k2, self.k1),
                           descr_mu, lap_mu, descr_comm_mu, 0,
                           descr1_red, descr2_red, list_descr, orient_op, ev_sqdiff)

            eval_native = opt_func.energy_func_std(*args_native)
            eval_orient = opt_func.oplist_commutation(np.eye(self.k2, self.k1), orient_op)
            orient_mu *= eval_native / eval_orient
            if verbose:
                print(f'\tScaling orientation preservation weight by {eval_native / eval_orient:.1e}')

        # Arguments for the optimization problem
        args = (descr_mu, lap_mu, descr_comm_mu, orient_mu,
                descr1_red, descr2_red, list_descr, orient_op, ev_sqdiff)

        # Initialization
        x0 = self.get_x0(optinit=optinit)

        if verbose:
            print(f'\nOptimization :\n'
                  f'\t{self.k1} Ev on source - {self.k2} Ev on Target\n'
                  f'\tUsing {self.descr1.shape[1]} Descriptors\n'
                  f'\tHyperparameters :\n'
                  f'\t\tDescriptors preservation :{descr_mu:.1e}\n'
                  f'\t\tDescriptors commutativity :{descr_comm_mu:.1e}\n'
                  f'\t\tLaplacian commutativity :{lap_mu:.1e}\n'
                  f'\t\tOrientation preservation :{orient_mu:.1e}\n'
                  )

        # Optimization
        start_time = time.time()
        res = fmin_l_bfgs_b(opt_func.energy_func_std, x0.ravel(), fprime=opt_func.grad_energy_std, args=args)
        opt_time = time.time() - start_time
        self.FM = res[0].reshape((self.k2, self.k1))

        if verbose:
            print("\tTask : {task}, funcall : {funcalls}, nit : {nit}, warnflag : {warnflag}".format(**res[2]))
            print(f'\tDone in {opt_time:.2f} seconds')

    def icp_refine(self, nit=10, tol=None, use_adj=False, overwrite=True, verbose=False):
        """
        Refines the functional map using ICP and saves the result

        Parameters
        -------------------
        nit       : int - number of iterations of icp to apply
        tol       : float - threshold of change in functional map in order to stop refinement
                    (only applies if nit is None)
        overwrite : bool - If True changes FM type to 'icp' so that next call of self.FM
                    will be the icp refined FM
        """
        if not self.fitted:
            raise ValueError("The Functional map must be fit before refining it")

        self.FM_icp = refine.mesh_icp_refine(self.eigenvectors1, self.eigenvectors2, self.FM,
                                             nit=nit, tol=tol, use_adj=use_adj, verbose=verbose)
        if overwrite:
            self.FM_type = 'icp'
        return self

    def zoomout_refine(self, nit=10, step=1, subsample=None, use_ANN=False, overwrite=True, verbose=False):
        """
        Refines the functional map using ZoomOut and saves the result

        Parameters
        -------------------
        nit       : int - number of iterations to do
        step      : increase in dimension at each Zoomout Iteration
        subsample : int - number of points to subsample for ZoomOut. If None or 0, no subsampling is done.
        use_ANN   : bool - If True, use approximate nearest neighbor
        overwrite : bool - If True changes FM type to 'zoomout' so that next call of self.FM
                    will be the zoomout refined FM (larger than the other 2)
        """
        if not self.fitted:
            raise ValueError("The Functional map must be fit before refining it")

        # if subsample is None or subsample == 0:
        #     sub = None
        # else:
        #     sub1 = self.mesh1.extract_fps(subsample)
        #     sub2 = self.mesh2.extract_fps(subsample)
        #     sub = (sub1, sub2)

        self.FM_zo = refine.mesh_zoomout_refine(self.eigenvectors1, self.eigenvectors2, self.A2, self.FM, nit,
                                                step=step, subsample=None, use_ANN=use_ANN, verbose=verbose)
        if overwrite:
            self.FM_type = 'zoomout'
        return self

    def compute_SD(self):
        """
        Compute the shape difference operators associated to the functional map
        """
        if not self.fitted:
            raise ValueError("The Functional map must be fit before computing the shape difference")

        self.D_a = spectral.area_SD(self.FM)
        self.D_c = spectral.conformal_SD(self.FM, self.mesh1.eigenvalues, self.mesh2.eigenvalues)

    def get_x0(self, optinit="zeros"):
        """
        Returns the initial functional map for optimization.

        Parameters
        ------------------------
        optinit : 'random' | 'identity' | 'zeros' initialization.
                  In any case, the first column of the functional map is computed by hand
                  and not modified during optimization

        Output
        ------------------------
        x0 : corresponding initial vector
        """
        if optinit == 'random':
            x0 = np.random.random((self.k2, self.k1))
        elif optinit == 'identity':
            x0 = np.eye(self.k2, self.k1)
        else:
            x0 = np.zeros((self.k2, self.k1))

        # Sets the equivalence between the constant functions
        ev_sign = np.sign(self.eigenvectors1[0, 0] * self.eigenvectors2[0, 0])
        area_ratio = np.sqrt(self.mesh2.get_surface_area() / self.mesh1.get_surface_area())

        x0[:, 0] = np.zeros(self.k2)
        x0[0, 0] = ev_sign * area_ratio

        return x0

    def compute_descr_op(self):
        """
        Compute the multiplication operators associated with the signatures

        Output
        ---------------------------
        operators : n_descr long list of ((k1,k1),(k2,k2)) operators.
        """
        if not self.preprocessed:
            raise ValueError("Preprocessing must be done before computing the new signatures")

        pinv1 = self.eigenvectors1[:, :self.k1].T @ self.A1  # (k1,n)
        pinv2 = self.eigenvectors2[:, :self.k2].T @ self.A2  # (k2,n)

        list_descr = [
            (pinv1 @ (self.descr1[:, i, None] * self.eigenvectors1[:, :self.k1]),
             pinv2 @ (self.descr2[:, i, None] * self.eigenvectors2[:, :self.k2])
             )
            for i in range(self.descr1.shape[1])
        ]

        return list_descr

    def compute_orientation_op(self, reversing=False, normalize=False):
        """
        Compute orientation preserving or reversing operators associated to each descriptor.

        Parameters
        ---------------------------------
        reversing : whether to return operators associated to orientation inversion instead
                    of orientation preservation (return the opposite of the second operator)
        normalize : whether to normalize the gradient on each face. Might improve results
                    according to the authors

        Output
        ---------------------------------
        list_op : (n_descr,) where term i contains (D1,D2) respectively of size (k1,k1) and
                  (k2,k2) which represent operators supposed to commute.
        """
        n_descr = self.descr1.shape[1]

        # Precompute the inverse of the eigenvectors matrix
        pinv1 = self.eigenvectors1[:, :self.k1].T @ self.A1  # (k1,n)
        pinv2 = self.eigenvectors2[:, :self.k2].T @ self.A2  # (k2,n)

        # Compute the gradient of each descriptor
        grads1 = [self.gradient(self.mesh1, self.descr1[:, i], normalize=normalize) for i in range(n_descr)]
        grads2 = [self.gradient(self.mesh2, self.descr2[:, i], normalize=normalize) for i in range(n_descr)]

        # Compute the operators in reduced basis
        can_op1 = [pinv1 @ self.orientation_op(self.mesh1, self.A1, gradf) @ self.eigenvectors1[:, :self.k1]
                   for gradf in grads1]

        if reversing:
            can_op2 = [- pinv2 @ self.orientation_op(self.mesh2, self.A2, gradf) @ self.eigenvectors2[:, :self.k2]
                       for gradf in grads2]
        else:
            can_op2 = [pinv2 @ self.orientation_op(self.mesh2, self.A2, gradf) @ self.eigenvectors2[:, :self.k2]
                       for gradf in grads2]

        list_op = list(zip(can_op1, can_op2))
        return list_op

    def gradient(self, mesh, f, normalize=False):
        """
        computes the gradient of a function on f using linear
        interpolation between vertices.

        Parameters
        --------------------------
        f         : (n_v,) function value on each vertex
        normalize : bool - Whether the gradient should be normalized on each face

        Output
        --------------------------
        gradient : (n_f,3) gradient of f on each face
        """
        vertlist = np.asarray(mesh.vertices)
        facelist = np.asarray(mesh.triangles)
        normals = np.asarray(mesh.triangle_normals)
        grad = geom.grad_f(f, vertlist, facelist, normals)  # (n_f,3)
        if normalize:
            grad /= np.linalg.norm(grad,axis=1,keepdims=True)
        return grad

    def orientation_op(self, mesh, A, gradf, normalize=False):
        """
        Compute the orientation operator associated to a gradient field gradf.

        For a given function g on the vertices, this operator linearly computes
        < grad(f) x grad(g), n> for each vertex by averaging along the adjacent faces.
        In practice, we compute < n x grad(f), grad(g) > for simpler computation.

        Parameters
        --------------------------
        gradf     : (n_f,3) gradient field on the mesh
        normalize : Whether to normalize the gradient on each face

        Output
        --------------------------
        operator : (n_v,n_v) orientation operator.
        """
        vertlist = np.asarray(mesh.vertices)
        facelist = np.asarray(mesh.triangles)
        normals = np.asarray(mesh.triangle_normals)
        if normalize:
            gradf /= np.linalg.norm(gradf, axis=1, keepdims=True)
        per_vert_area = np.asarray(A.sum(1)).flatten()
        operator = geom.get_orientation_op(gradf, vertlist, facelist, normals,
                                           per_vert_area)
        return operator

    def project(self, func, k=None, mesh_ind=1):
        """
        Projects a function on the LB basis

        Parameters
        -----------------------
        func    : array - (n1|n2,p) evaluation of the function
        mesh_in : int  1 | 2 index of the mesh on which to encode

        Output
        -----------------------
        encoded_func : (n1|n2,p) array of decoded f
        """
        if k is None:
            k = self.k1 if mesh_ind == 1 else self.k2

        if mesh_ind == 1:
            return self.eigenvectors1[:, :k].T @ self.A1 @ func
        elif mesh_ind == 2:
            return self.eigenvectors2[:, :k].T @ self.A2 @ func
        else:
            raise ValueError(f'Only indices 1 or 2 are accepted, not {mesh_ind}')

    def decode(self, encoded_func, mesh_ind=2):
        """
        Decode a function from the LB basis

        Parameters
        -----------------------
        encoded_func : array - (k1|k2,p) encoding of the functions
        mesh_ind     : int  1 | 2 index of the mesh on which to decode

        Output
        -----------------------
        func : (n1|n2,p) array of decoded f
        """

        if mesh_ind == 1:
            return self.mesh1.decode(encoded_func)
        elif mesh_ind == 2:
            return self.mesh2.decode(encoded_func)
        else:
            raise ValueError(f'Only indices 1 or 2 are accepted, not {mesh_ind}')

    def transport(self, encoded_func, reverse=False):
        """
        transport a function from LB basis 1 to LB basis 2.
        If reverse is True, then the functions are transposed the other way
        using the transpose of the functional map matrix

        Parameters
        -----------------------
        encoded_func : array - (k1|k2,p) encoding of the functions
        reverse      : bool If true, transpose from 2 to 1 using the transpose of the FM

        Output
        -----------------------
        transp_func : (n2|n1,p) array of new encoding of the functions
        """
        if not self.preprocessed:
            raise ValueError("The Functional map must be fit before transporting a function")

        if not reverse:
            return self.FM @ encoded_func
        else:
            return self.FM.T @ encoded_func

    def transfer(self, func, reverse=False):
        """
        Transfer a function from mesh1 to mesh2.
        If 'reverse' is set to true, then the transfer goes
        the other way using the transpose of the functional
        map as approximate inverser transfer.

        Parameters
        ----------------------
        func : (n1|n2,p) evaluation of the functons

        Output
        -----------------------
        transp_func : (n2|n1,p) transfered function

        """
        if not reverse:
            return self.decode(self.transport(self.project(func)))

        else:
            encoding = self.project(func, mesh_ind=2)
            return self.decode(self.transport(encoding, reverse=True),
                               mesh_ind=1
                               )
