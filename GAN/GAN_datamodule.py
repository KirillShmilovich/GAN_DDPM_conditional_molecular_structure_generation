from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, TensorDataset
from abc import abstractmethod

import mdtraj as md
import mdshare
from scipy.spatial.distance import pdist
from scipy.stats import pearsonr
import numpy as np
from pyemma.coordinates import tica
from sklearn.preprocessing import MinMaxScaler
import torch


class GANBaseDataModule(LightningDataModule):
    def __init__(self, batch_size: int = 1000):
        super().__init__()
        self.get_data()
        self.batch_size = batch_size

        self.x_dim = self.train_data[0].shape[1]
        self.c_dim = self.train_data[1].shape[1]
        self.n_samples = self.eval_data[0].shape[0]

    @abstractmethod
    def get_data(self):
        pass

    @abstractmethod
    def get_eval_metrics(self):
        pass

    def train_dataloader(self):
        dataset = TensorDataset(*self.train_data)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=True)


class PentaPeptideBackbone(GANBaseDataModule):
    def get_data(self):
        N_TRAIN_TRJS = 20
        CONDITIONING_DIM = 4

        pdb = mdshare.fetch(
            "pentapeptide-impl-solv.pdb", working_directory="data_mdshare"
        )
        files = mdshare.fetch(
            "pentapeptide-*-500ns-impl-solv.xtc", working_directory="data_mdshare"
        )
        trjs = [md.load(file, top=pdb).center_coordinates() for file in files]
        train_trj = md.join(trjs[:N_TRAIN_TRJS])
        val_trj = md.join(trjs[N_TRAIN_TRJS:])
        print(
            f"Using {len(trjs[:N_TRAIN_TRJS])} trajectories for training consisting of {train_trj.n_frames} frames"
        )
        print(
            f"Using {len(trjs[N_TRAIN_TRJS:])} trajectories for evaluation consisting of {val_trj.n_frames} frames"
        )

        cond_idxs = train_trj.top.select("backbone")
        train_cond_trj = train_trj.atom_slice(cond_idxs).center_coordinates()
        val_cond_trj = val_trj.atom_slice(cond_idxs).center_coordinates()

        pdists_train = [
            np.concatenate([pdist(xyz)[None] for xyz in trj.atom_slice(cond_idxs).xyz])
            for trj in trjs[:N_TRAIN_TRJS]
        ]
        pdists_val = [
            np.concatenate([pdist(xyz)[None] for xyz in trj.atom_slice(cond_idxs).xyz])
            for trj in trjs[N_TRAIN_TRJS:]
        ]

        print(f"Generating {CONDITIONING_DIM}-dim TICA embedding as the conditioning")
        TICA = tica(pdists_train, lag=5, dim=CONDITIONING_DIM)
        projected_data_train = np.array(TICA.transform(pdists_train))
        projected_data_val = np.array(TICA.transform(pdists_val))

        condition_train = projected_data_train.reshape(
            -1, projected_data_train.shape[-1]
        )
        xyz_train = train_cond_trj.xyz.reshape(condition_train.shape[0], -1)

        condition_val = projected_data_val.reshape(-1, projected_data_val.shape[-1])
        xyz_val = val_cond_trj.xyz.reshape(condition_val.shape[0], -1)

        xyz_scaler = MinMaxScaler((-1, 1))
        xyz_train_scaled = xyz_scaler.fit_transform(xyz_train)
        xyz_val_scaled = xyz_scaler.transform(xyz_val)

        cond_scaler = MinMaxScaler((-1, 1))
        condition_train_scaled = cond_scaler.fit_transform(condition_train)
        condition_val_scaled = cond_scaler.transform(condition_val)

        self.train_data = [
            torch.Tensor(xyz_train_scaled).float(),
            torch.Tensor(condition_train_scaled).float(),
        ]
        self.eval_data = [
            torch.Tensor(xyz_val_scaled).float(),
            torch.Tensor(condition_val_scaled).float(),
        ]
        self.xyz_scaler = xyz_scaler
        self.condition_scaler = cond_scaler
        self.tica_estimator = TICA

        self.val_trj = val_trj
        self.val_cond_trj = val_cond_trj
        self.projected_data_val = condition_val

    def get_eval_metrics(self, pos_val):
        trj_val = md.Trajectory(pos_val, topology=self.val_cond_trj.top)
        pdists = np.concatenate([pdist(xyz)[None] for xyz in trj_val.xyz])
        projected_data = self.tica_estimator.transform(pdists)

        eval_metrics_dict = {}

        # Bond RMSDs
        trj_ref = self.val_cond_trj
        all_bonds = [(b[0].index, b[1].index) for b in trj_ref.top.bonds]
        bond_dists_r = md.compute_distances(trj_ref, all_bonds)
        bond_dists_s = md.compute_distances(trj_val, all_bonds)
        bond_rmsd = np.sqrt(np.mean((bond_dists_s - bond_dists_r) ** 2))
        eval_metrics_dict["bond_rmsd"] = bond_rmsd

        # TIC correlations
        projected_data_true = self.projected_data_val
        for i in range(projected_data.shape[1]):
            p_phi, _ = pearsonr(projected_data_true[:, i], projected_data[:, i])
            eval_metrics_dict[f"TIC_correlations/TIC_{i}"] = p_phi

        # AUCs
        min_r, max_r = np.min(bond_dists_r, axis=0), np.max(bond_dists_r, axis=0)
        correct_aa = np.sum((bond_dists_s < max_r) & (bond_dists_s > min_r), axis=1)
        auc_aa = []
        for i in range(len(all_bonds)):
            auc = np.mean(correct_aa > i)
            auc_aa.append(auc)
        eval_metrics_dict["AUC"] = np.sum(auc_aa) / len(all_bonds)
        return eval_metrics_dict, trj_val


class PentaPeptideHeavy(GANBaseDataModule):
    def get_data(self):
        N_TRAIN_TRJS = 20
        CONDITIONING_DIM = 4

        pdb = mdshare.fetch(
            "pentapeptide-impl-solv.pdb", working_directory="data_mdshare"
        )
        files = mdshare.fetch(
            "pentapeptide-*-500ns-impl-solv.xtc", working_directory="data_mdshare"
        )
        trjs = [md.load(file, top=pdb).center_coordinates() for file in files]
        train_trj = md.join(trjs[:N_TRAIN_TRJS])
        val_trj = md.join(trjs[N_TRAIN_TRJS:])
        print(
            f"Using {len(trjs[:N_TRAIN_TRJS])} trajectories for training consisting of {train_trj.n_frames} frames"
        )
        print(
            f"Using {len(trjs[N_TRAIN_TRJS:])} trajectories for evaluation consisting of {val_trj.n_frames} frames"
        )

        cond_idxs = train_trj.top.select_atom_indices("heavy")
        # train_cond_trj = train_trj.atom_slice(cond_idxs).center_coordinates()
        val_cond_trj = val_trj.atom_slice(cond_idxs).center_coordinates()

        pdists_train = [
            np.concatenate([pdist(xyz)[None] for xyz in trj.atom_slice(cond_idxs).xyz])
            for trj in trjs[:N_TRAIN_TRJS]
        ]
        pdists_val = [
            np.concatenate([pdist(xyz)[None] for xyz in trj.atom_slice(cond_idxs).xyz])
            for trj in trjs[N_TRAIN_TRJS:]
        ]

        print(f"Generating {CONDITIONING_DIM}-dim TICA embedding as the conditioning")
        TICA = tica(pdists_train, lag=5, dim=CONDITIONING_DIM)
        projected_data_train = np.array(TICA.transform(pdists_train))
        projected_data_val = np.array(TICA.transform(pdists_val))

        condition_train = projected_data_train.reshape(
            -1, projected_data_train.shape[-1]
        )
        xyz_train = train_trj.xyz.reshape(condition_train.shape[0], -1)

        condition_val = projected_data_val.reshape(-1, projected_data_val.shape[-1])
        xyz_val = val_trj.xyz.reshape(condition_val.shape[0], -1)

        xyz_scaler = MinMaxScaler((-1, 1))
        xyz_train_scaled = xyz_scaler.fit_transform(xyz_train)
        xyz_val_scaled = xyz_scaler.transform(xyz_val)

        cond_scaler = MinMaxScaler((-1, 1))
        condition_train_scaled = cond_scaler.fit_transform(condition_train)
        condition_val_scaled = cond_scaler.transform(condition_val)

        self.train_data = [
            torch.Tensor(xyz_train_scaled).float(),
            torch.Tensor(condition_train_scaled).float(),
        ]
        self.eval_data = [
            torch.Tensor(xyz_val_scaled).float(),
            torch.Tensor(condition_val_scaled).float(),
        ]
        self.xyz_scaler = xyz_scaler
        self.condition_scaler = cond_scaler
        self.tica_estimator = TICA
        self.cond_idxs = cond_idxs
        self.val_trj = val_trj
        self.val_cond_trj = val_cond_trj
        self.projected_data_val = condition_val

    def get_eval_metrics(self, pos_val):
        trj_val = md.Trajectory(pos_val, topology=self.val_trj.top)
        pdists = np.concatenate(
            [pdist(xyz)[None] for xyz in trj_val.atom_slice(self.cond_idxs).xyz]
        )
        projected_data = self.tica_estimator.transform(pdists)

        eval_metrics_dict = {}

        # Bond RMSDs
        trj_ref = self.val_trj
        all_bonds = [(b[0].index, b[1].index) for b in trj_ref.top.bonds]
        bond_dists_r = md.compute_distances(trj_ref, all_bonds)
        bond_dists_s = md.compute_distances(trj_val, all_bonds)
        bond_rmsd = np.sqrt(np.mean((bond_dists_s - bond_dists_r) ** 2))
        eval_metrics_dict["bond_rmsd"] = bond_rmsd

        # TIC correlations
        projected_data_true = self.projected_data_val
        for i in range(projected_data.shape[1]):
            p_phi, _ = pearsonr(projected_data_true[:, i], projected_data[:, i])
            eval_metrics_dict[f"TIC_correlations/TIC_{i}"] = p_phi

        # AUCs
        min_r, max_r = np.min(bond_dists_r, axis=0), np.max(bond_dists_r, axis=0)
        correct_aa = np.sum((bond_dists_s < max_r) & (bond_dists_s > min_r), axis=1)
        auc_aa = []
        for i in range(len(all_bonds)):
            auc = np.mean(correct_aa > i)
            auc_aa.append(auc)
        eval_metrics_dict["AUC"] = np.sum(auc_aa) / len(all_bonds)
        return eval_metrics_dict, trj_val


class AlanineDipeptide(GANBaseDataModule):
    def get_data(self):
        N_TRAIN_TRJS = 2
        # CONDITIONING_DIM = 2

        pdb = mdshare.fetch(
            "alanine-dipeptide-nowater.pdb", working_directory="data_mdshare"
        )
        files = mdshare.fetch(
            "alanine-dipeptide-*-250ns-nowater.xtc", working_directory="data_mdshare"
        )
        trjs = [md.load(file, top=pdb).center_coordinates() for file in files]
        train_trj = md.join(trjs[:N_TRAIN_TRJS])
        val_trj = md.join(trjs[N_TRAIN_TRJS:])
        print(
            f"Using {len(trjs[:N_TRAIN_TRJS])} trajectories for training consisting of {train_trj.n_frames} frames"
        )
        print(
            f"Using {len(trjs[N_TRAIN_TRJS:])} trajectories for evaluation consisting of {val_trj.n_frames} frames"
        )

        cond_idxs = train_trj.top.select("backbone")
        train_cond_trj = train_trj.atom_slice(cond_idxs).center_coordinates()
        val_cond_trj = val_trj.atom_slice(cond_idxs).center_coordinates()

        _, phi_train = md.compute_phi(train_trj)
        _, psi_train = md.compute_psi(train_trj)
        condition_train = np.concatenate((phi_train, psi_train), -1)

        _, phi_val = md.compute_phi(val_trj)
        _, psi_val = md.compute_psi(val_trj)
        condition_val = np.concatenate((phi_val, psi_val), -1)

        xyz_train = train_cond_trj.xyz.reshape(condition_train.shape[0], -1)
        xyz_val = val_cond_trj.xyz.reshape(condition_val.shape[0], -1)

        xyz_scaler = MinMaxScaler((-1, 1))
        xyz_train_scaled = xyz_scaler.fit_transform(xyz_train)
        xyz_val_scaled = xyz_scaler.transform(xyz_val)

        cond_scaler = MinMaxScaler((-1, 1))
        condition_train_scaled = cond_scaler.fit_transform(condition_train)
        condition_val_scaled = cond_scaler.transform(condition_val)

        self.train_data = [
            torch.Tensor(xyz_train_scaled).float(),
            torch.Tensor(condition_train_scaled).float(),
        ]
        self.eval_data = [
            torch.Tensor(xyz_val_scaled).float(),
            torch.Tensor(condition_val_scaled).float(),
        ]
        self.xyz_scaler = xyz_scaler
        self.condition_scaler = cond_scaler
        self.cond_idxs = cond_idxs
        self.val_trj = val_trj
        self.val_cond_trj = val_cond_trj
        self.projected_data_val = condition_val

    def get_eval_metrics(self, pos_val):
        trj_val = md.Trajectory(pos_val, topology=self.val_cond_trj.top)
        _, phi_val = md.compute_phi(trj_val)
        _, psi_val = md.compute_psi(trj_val)
        projected_data = np.concatenate((phi_val, psi_val), -1)

        eval_metrics_dict = {}

        # Bond RMSDs
        trj_ref = self.val_cond_trj
        all_bonds = [(b[0].index, b[1].index) for b in trj_ref.top.bonds]
        bond_dists_r = md.compute_distances(trj_ref, all_bonds)
        bond_dists_s = md.compute_distances(trj_val, all_bonds)
        bond_rmsd = np.sqrt(np.mean((bond_dists_s - bond_dists_r) ** 2))
        eval_metrics_dict["bond_rmsd"] = bond_rmsd

        # phi,psi correlations
        projected_data_true = self.projected_data_val
        for i in range(projected_data.shape[1]):
            p_phi, _ = pearsonr(projected_data_true[:, i], projected_data[:, i])
            if i == 0:
                eval_metrics_dict["Phi_Psi_correlations/Phi"] = p_phi
            if i == 1:
                eval_metrics_dict["Phi_Psi_correlations/Psi"] = p_phi

        # AUCs
        min_r, max_r = np.min(bond_dists_r, axis=0), np.max(bond_dists_r, axis=0)
        correct_aa = np.sum((bond_dists_s < max_r) & (bond_dists_s > min_r), axis=1)
        auc_aa = []
        for i in range(len(all_bonds)):
            auc = np.mean(correct_aa > i)
            auc_aa.append(auc)
        eval_metrics_dict["AUC"] = np.sum(auc_aa) / len(all_bonds)
        return eval_metrics_dict, trj_val
