import matplotlib.pyplot as plt
import numpy as np
import numpy.matlib as mat
import tensorflow as tf
from scipy.interpolate import interp1d


class ProMP:
    """A simplified implementation of ProMP.

    Original paper: A. Paraschos, C. Daniel, J. Peters, and G. Neumann, ‘Probabilistic Movement Primitives’, in Proceedings of the 26th International
    Conference on Neural Information Processing Systems - Volume 2, 2013, pp. 2616–2624.
    """

    def __init__(self, n_basis: int, n_dof: int, n_t: int, h: float = 0.07, f: float = 1.0):
        """A simplified implementation of ProMP.

        Original paper: A. Paraschos, C. Daniel, J. Peters, and G. Neumann, ‘Probabilistic Movement Primitives’, in Proceedings of the 26th International
        Conference on Neural Information Processing Systems - Volume 2, 2013, pp. 2616–2624.

        Args:
            n_basis (int): Number of basis functions.
            n_dof (int): Number of joints.
            n_t (int): Number of discrete time points.
            h (float, optional): Bandwidth of the basis functions. Defaults to 0.07.
            f (int, optional): Modulation factor. Defaults to 1.
        """
        self.n_basis = n_basis
        self.n_dof = n_dof
        self.n_t = n_t
        # Time step.
        self.dt = 1 / (n_t - 1)
        self.h = h
        self.f = f

        # The block-diagonal matrix for all DOFs.
        self.all_phi = np.kron(np.eye(self.n_dof, dtype=int), self.basis_func_gauss_glb())  # (T * n_dof, n * n_dof)
        self.all_phi_t = np.transpose(self.all_phi)
        # Tensor versions to be used in metrics and losses.
        # The cast to float32 is required since other values in the program use float32 and default is float64.
        self.all_phi_tf = tf.cast(self.all_phi, tf.float32)
        self.all_phi_t_tf = tf.cast(self.all_phi_t, tf.float32)

    def basis_func_gauss_glb(self) -> np.ndarray:
        """Evaluates Gaussian basis functions in [0,1].

        This is used globally in the loss function.

        Returns:
            np.ndarray: The basis functions phi with shape (T, n_basis).
        """

        tf_ = 1/self.f
        T = int(round(tf_/self.dt+1))
        F = np.zeros((T, self.n_basis))
        for z in range(0, T):
            t = z*self.dt
            q = np.zeros((1, self.n_basis))
            for k in range(1, self.n_basis + 1):
                c = (k - 1) / (self.n_basis - 1)
                q[0, k - 1] = np.exp(-(self.f * t - c) * (self.f * t - c) / (2 * self.h))
            F[z, :self.n_basis] = q[0, :self.n_basis]

        # Normalize basis functions
        F = F / np.transpose(mat.repmat(np.sum(F, axis=1), self.n_basis, 1))
        # The output has shape (T, n_basis).
        return F

    def basis_func_gauss_local(self, T: int) -> np.ndarray:
        """Evaluates Gaussian basis functions in [0,1].

        This is used for each trajectory.

        Args:
            T (int): Number of discrete time instants.

        Returns:
            np.ndarray: The basis functions phi with shape (T, n_basis).
        """

        dt = 1/(T-1)
        F = np.zeros((T, self.n_basis))
        for z in range(0, T):
            t = z*dt
            q = np.zeros((1, self.n_basis))
            for k in range(1, self.n_basis + 1):
                c = (k - 1) / (self.n_basis - 1)
                q[0, k - 1] = np.exp(-(self.f * t - c) * (self.f * t - c)/(2 * self.h))
            F[z, :self.n_basis] = q[0, :self.n_basis]

        # Normalize the basis functions.
        F = F / np.transpose(mat.repmat(np.sum(F, axis=1), self.n_basis, 1))
        # The output has shape (T, n_basis).
        return F

    def weights_from_trajectory(self, trajectory: np.ndarray, vector_output: bool = True) -> np.ndarray:
        """Calculate the weights corresponding to a trajectory.

        Only the expected value is calculated.

        Args:
            trajectory (np.ndarray): Time history of each dof with shape (samples, n_dof).
            vector_output (bool, optional): If True the output is given in vector shape (n_dof * n_basis, ). \
                                            If False it is given in matrix shape (n_dof, n_basis). \
                                            Defaults to True.

        Returns:
            np.ndarray: The ProMP weights in a (n_dof * n_basis, ) vector or in a (n_dof, n_basis) matrix.
        """
        num_samples = trajectory.shape[0]
        phi = self.basis_func_gauss_local(num_samples)  # (n_samples, n_basis)
        weights = np.transpose(np.matmul(np.linalg.pinv(phi), trajectory))  # (n_dof, n_basis)

        if vector_output:
            # Reshape matrix as vector.
            return weights.reshape((-1, ))  # (n_basis * n_dof, )
        else:
            # Keep matrix shape.
            return weights  # (n_dof, n_basis)

    def trajectory_from_weights(self, weights: np.ndarray, vector_output: bool = False) -> np.ndarray:
        """Calculate the trajectory of all DOFs from the given weights.

        Args:
            weights (np.ndarray): The ProMP weights with shape (n_basis * n_dof, ).
            vector_output (bool, optional): If True the output is given in vector shape (n_t * n_dof, ). \
                                            If False it is given in matrix shape (n_t, n_dof). \
                                            Defaults to False.

        Returns:
            np.ndarray: The trajectories of all DOFs in a (n_t, n_dof) matrix or in a (n_t * n_dof, ) vector.
        """
        trajectory = np.matmul(weights, self.all_phi_t)
        if vector_output:
            # Keep vector shape.
            return trajectory  # (n_t * n_dof, )
        else:
            # Reshape into a matrix.
            return np.transpose(np.reshape(trajectory, (self.n_dof, self.n_t)))  # (n_t, n_dof)

    def trajectory_from_weights_tf(self, weights: tf.Tensor, vector_output: bool = False) -> tf.Tensor:
        """Calculate the trajectory of all DOFs from the given weights, using TensorFlow tensors and operations.

        Args:
            weights (tf.Tensor): The ProMP weights with shape (n_batch, n_basis * n_dof, ).
            vector_output (bool, optional): If True the output is given in vector shape (n_batch, n_t * n_dof, ).\n \
                                            If False it is given in matrix shape (n_batch, n_t, n_dof). Defaults to False.

        Returns:
            tf.Tensor: The trajectories of all DOFs in a (n_batch, n_t, n_dof) matrix or in a (n_batch, n_t * n_dof, ) vector.
        """
        trajectory = tf.matmul(weights, self.all_phi_t_tf)
        if vector_output:
            # Keep vector shape.
            return trajectory  # (n_batch, n_t * n_dof, )
        else:
            # Reshape into a matrix.
            return tf.transpose(tf.reshape(trajectory, (-1, self.n_dof, self.n_t)), (0, 2, 1))  # (n_batch, n_t, n_dof)


class ProMPTuner:

    def __init__(self, trajectories: list[np.ndarray], promp: ProMP) -> None:
        self.promp = promp
        self.trajectories = trajectories  # [(n_samples, n_dof)]

        # Linearly interpolate the trajectories to match the output of the ProMP.
        self.trajectories_interpolated = []
        for traj in self.trajectories:
            traj_interpolator = interp1d(np.linspace(0, 1, traj.shape[0]), traj, axis=0)
            self.trajectories_interpolated.append(traj_interpolator(np.linspace(0, 1, promp.n_t)))

    def tune_n_basis(self, min: int = 2, max: int = 10, step: int = 1, show: bool = False) -> tuple[np.ndarray, np.ndarray]:
        """Tune the n_basis parameter of a ProMP.

        The tuning is performed by computing the reconstruction MSE on the trajectories provided for different n_basis parameters.

        Args:
            min (int, optional): Minimum number of basis. Defaults to 2.
            max (int, optional): Maximum number of basis. Defaults to 10.
            step (int, optional): Step in the number of basis. Defaults to 1.
            show (bool, optional): If True, show the text and plot with the results. Defaults to False.

        Returns:
            tuple[np.ndarray, np.ndarray]: The average trajectory reconstruction mean square errors and the correpsonding list of n_basis parameters.
        """
        assert 2 <= min <= max, "'min' should be between 2 and 'max'"
        assert step > 0, "'step' should be > 0"

        n_traj = len(self.trajectories)
        n_basis_options = np.array(range(min, max+1, step))
        mse = np.zeros_like(n_basis_options, dtype=float)
        for i, n_basis in enumerate(n_basis_options):
            promp = ProMP(n_basis, n_dof=self.promp.n_dof, n_t=self.promp.n_t, h=self.promp.h, f=self.promp.f)
            for traj in self.trajectories_interpolated:
                traj_interpolator = interp1d(np.linspace(0, 1, traj.shape[0]), traj, axis=0)
                traj_interpolated = traj_interpolator(np.linspace(0, 1, promp.n_t))
                traj_rec = promp.trajectory_from_weights(promp.weights_from_trajectory(traj_interpolated))
                mse[i] += np.mean((traj_interpolated - traj_rec)**2)
            mse[i] /= n_traj

        if show:
            print("n_basis: mse(trajectory)")
            for n_basis, mse_val in zip(n_basis_options, mse):
                print(f"{n_basis}: {mse_val:.3e}")
            plt.plot(n_basis_options, mse, 'o-')

            plt.grid(True)
            plt.show()

        return mse, n_basis_options


if __name__ == "__main__":
    import numpy as np

    N_DOF = 3
    N_BASIS = 5
    N_T = 100

    def random_polynomial(t: np.ndarray, n_zeros: int = 1, scale: float = 1.0, y_offset: float = 0.0) -> np.ndarray:
        """Sample a random polynomial function.

        The polynomial will have n_zeros zeros uniformly sampled between min(t) and max(t).
        By default the polynomial will have co-domain [0, 1], but this can be changed with the scale and y_offset arguments.

        Args:
            t (np.ndarray): Time vector.
            n_zeros (int, optional): Number of zeros of the polynomial. Defaults to 1.
            scale (float, optional): Scale of the polynomial. Defaults to 1.0.
            y_offset (float, optional): Offset from y=0. Defaults to 0.0.

        Returns:
            np.ndarray: The polynomial sampled on t.
        """
        zeros = np.random.uniform(np.min(t), np.max(t), n_zeros)
        y = np.ones_like(t)
        for t_0 in zeros:
            y *= t - t_0
        y_min = np.min(y)
        y_max = np.max(y)
        return y_offset + (y - y_min) / (y_max - y_min) * scale

    # Generate random trajectories.
    t = np.linspace(0, 1, N_T)
    traj = np.zeros((N_T, N_DOF))
    for i in range(N_DOF):
        traj[:, i] = random_polynomial(t, n_zeros=4, scale=5)

    # Initialize the ProMP.
    promp = ProMP(N_BASIS, N_DOF, N_T)
    # Compute ProMP weights.
    promp_weights = promp.weights_from_trajectory(traj)
    # Reconstruct the trajectories.
    traj_rec = promp.trajectory_from_weights(promp_weights)

    # Show a comparison between original and reconstructed trajectories.
    fig, axs = plt.subplots(N_DOF, 1)
    for i in range(N_DOF):
        axs[i].plot(t, traj[:, i], t, traj_rec[:, i])
        axs[i].legend(("Original", "Reconstructed"))
    plt.show()

    # Tune the n_basis parameter.
    promp_tuner = ProMPTuner(np.expand_dims(traj, axis=0), promp)
    promp_tuner.tune_n_basis(show=True)
