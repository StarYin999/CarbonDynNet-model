import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class WarmupCosineScheduler:
    def __init__(self, optimizer, warmup_epochs, total_epochs, min_lr=1e-6, max_lr=0.01):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.current_epoch = 0
    def step(self):
        self.current_epoch += 1
        if self.current_epoch <= self.warmup_epochs:
            lr = self.min_lr + (self.max_lr - self.min_lr) * (self.current_epoch / self.warmup_epochs)
        else:
            progress = (self.current_epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            lr = self.min_lr + 0.5 * (self.max_lr - self.min_lr) * (1 + math.cos(math.pi * progress))
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        return lr

class MonodKineticsLayer(nn.Module):
    def __init__(self, hidden_dim, num_nodes):
        super(MonodKineticsLayer, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_nodes = num_nodes
        self.substrates = {'SF': 0, 'Spro': 1, 'Sac': 2, 'SH': 3, 'SCH4': 4, 'SSO4': 5, 'SH2S': 6, 'Se': 7, 'SNH4': 8, 'SNO3': 9, 'SNO2': 10, 'SN2O': 11, 'SN2': 12, 'SFe3': 13, 'SFe2': 14, 'SFeS': 15, 'XS': 16, 'XI': 17}
        self.microbes = {'Xacid': 18, 'Xacet': 19, 'Xmeth_ac': 20, 'Xmeth_h': 21, 'XSRB_pro': 22, 'XSRB_ac': 23, 'XSRB_h': 24, 'XAOB': 25, 'XNOB': 26, 'XDNB': 27, 'XFeRB': 28}
        self.env_params = {'flow_rate': 29, 'flow_volume': 30, 'pH': 31, 'ORP': 32}
        self.mu_max = nn.ParameterDict({k: nn.Parameter(torch.rand(num_nodes) * 0.015 + 0.005) for k in self.substrates})
        self.Ks = nn.ParameterDict({k: nn.Parameter(torch.rand(num_nodes) * 0.3 + 0.2) for k in self.substrates})
        self.env_impact = nn.ParameterDict({f'{substrate}_{env}': nn.Parameter(torch.randn(num_nodes) * 0.001) for substrate in self.substrates for env in self.env_params})
        self.k_sed_hyd = nn.Parameter(torch.rand(num_nodes) * 0.01 + 0.001)
        self.v_crit = nn.Parameter(torch.rand(num_nodes) * 0.2 + 0.2)
        self.flow_sensitivity = nn.Parameter(torch.ones(num_nodes) * 5.0)
        self.sediment_yields = nn.Parameter(torch.rand(num_nodes, 3))
        self.reactions = {
            'hydrolysis': {'substrates': ['SF', 'XS'], 'products': ['Sac', 'Spro'], 'microbes': ['Xacid', 'Xacet'], 'type': 'monod_inhibition'},
            'methanogenesis': {'substrates': ['Sac', 'SH'], 'products': ['SCH4'], 'microbes': ['Xmeth_ac', 'Xmeth_h'], 'type': 'monod'},
            'sulfate_reduction': {'substrates': ['Spro', 'Sac', 'SH', 'SSO4'], 'products': ['SH2S'], 'microbes': ['XSRB_pro', 'XSRB_ac', 'XSRB_h'], 'type': 'monod'},
            'aerobic_respiration': {'substrates': ['SF', 'Sac', 'Se'], 'products': ['SNO3'], 'microbes': ['XDNB'], 'type': 'monod'},
            'nitrification': {'substrates': ['SNH4', 'SNO2', 'Se'], 'products': ['SNO2', 'SNO3'], 'microbes': ['XAOB', 'XNOB'], 'type': 'monod'},
            'denitrification': {'substrates': ['SF', 'SNO3', 'SNO2', 'SN2O'], 'products': ['SN2O', 'SN2'], 'microbes': ['XDNB'], 'type': 'monod'},
            'iron_reduction': {'substrates': ['SF', 'SFe3'], 'products': ['SFe2'], 'microbes': ['XFeRB'], 'type': 'monod'},
            'iron_sulfide_formation': {'substrates': ['SH2S', 'SFe2'], 'products': ['SFeS'], 'microbes': [], 'type': 'monod'},
            'sediment_process': {'substrates': ['XS'], 'products': ['SF', 'Sac', 'SNH4'], 'microbes': [], 'type': 'sedimentation'}
        }
        self.transform = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
    def monod_equation(self, S, substrate_name, env_params):
        mu_raw = self.mu_max[substrate_name].view(-1, 1)
        Ks_raw = self.Ks[substrate_name].view(-1, 1)
        epsilon = 1e-9
        mu = torch.clamp(F.softplus(mu_raw), min=epsilon)
        Ks = torch.clamp(F.softplus(Ks_raw), min=epsilon)
        env_impact = torch.zeros_like(S)
        for env_name, env_idx in self.env_params.items():
            env_value = env_params[:, :, env_idx]
            env_weight = self.env_impact[f'{substrate_name}_{env_name}'].view(-1, 1)
            env_impact += env_value * env_weight
        env_impact = torch.clamp(env_impact, -1, 1)
        mu_effective = mu * (1 + env_impact)
        Ks_effective = Ks * (1 + env_impact)
        if substrate_name in ['SF', 'XS', 'Spro']:
            Ki = 30.0
            inhibition_factor = 1.0 / (1.0 + S / Ki)
            return mu_effective * S / (Ks_effective + S + 1e-6) * inhibition_factor
        else:
            return mu_effective * S / (Ks_effective + S + 1e-6)
    def sedimentation_equation(self, S_xs: torch.Tensor, flow_v: torch.Tensor) -> torch.Tensor:
        epsilon = 1e-9
        k_sed = torch.clamp(F.softplus(self.k_sed_hyd).view(-1, 1), min=epsilon)
        v_crit = torch.clamp(F.softplus(self.v_crit).view(-1, 1), min=epsilon)
        sens = torch.clamp(F.softplus(self.flow_sensitivity).view(-1, 1), min=epsilon)
        flow_factor = torch.sigmoid(-sens * (flow_v - v_crit))
        sed_rate = k_sed * S_xs * flow_factor
        return torch.clamp(sed_rate, min=0.0) + 1e-9
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        output = torch.zeros_like(x)
        calculated_reaction_rates = {}
        for reaction_name, reaction_info in self.reactions.items():
            substrates = reaction_info['substrates']
            products = reaction_info['products']
            microbes = reaction_info.get('microbes', [])
            r_type = reaction_info['type']
            if r_type == 'sedimentation':
                xs_idx = self.substrates.get('XS', -1)
                flow_idx = self.env_params.get('flow_rate', -1)
                if xs_idx == -1 or xs_idx >= x.shape[2] or flow_idx == -1 or flow_idx >= x.shape[2]:
                    continue
                XS = x[:, :, xs_idx]
                flow = x[:, :, flow_idx]
                reaction_rate = self.sedimentation_equation(XS, flow)
                calculated_reaction_rates[reaction_name] = reaction_rate
                output[:, :, xs_idx] -= reaction_rate
                yields_raw = self.sediment_yields
                yields_frac = torch.softmax(yields_raw, dim=1)
                y_sf = yields_frac[:, 0].view(-1, 1)
                y_sac = yields_frac[:, 1].view(-1, 1)
                y_snh4 = yields_frac[:, 2].view(-1, 1)
                for prod_key, y_node in zip(['SF', 'Sac', 'SNH4'], [y_sf, y_sac, y_snh4]):
                    if prod_key in self.substrates:
                        p_idx = self.substrates[prod_key]
                        if p_idx < x.shape[2]:
                            output[:, :, p_idx] += torch.clamp(reaction_rate * y_node, min=0.0)
                continue
            growth_rates_for_reaction = []
            for substrate_key in substrates:
                if substrate_key not in self.substrates:
                    continue
                substrate_idx = self.substrates[substrate_key]
                if substrate_idx >= x.shape[2]:
                    growth_rates_for_reaction.append(torch.full_like(x[:, :, 0], -float('inf')))
                    continue
                S = x[:, :, substrate_idx]
                rate_term_for_substrate = self.monod_equation(S, substrate_key, x)
                growth_rates_for_reaction.append(rate_term_for_substrate)
            if not growth_rates_for_reaction:
                growth_rate = torch.zeros_like(x[:, :, 0])
            else:
                growth_rate = torch.min(torch.stack(growth_rates_for_reaction, dim=0), dim=0)[0]
            microbe_conc = torch.zeros_like(growth_rate)
            if microbes:
                for microbe_key in microbes:
                    if microbe_key not in self.microbes:
                        continue
                    microbe_idx = self.microbes[microbe_key]
                    if microbe_idx >= x.shape[2]:
                        continue
                    microbe_conc += x[:, :, microbe_idx]
            reaction_rate = torch.clamp(growth_rate * microbe_conc, min=0.0) + 1e-9
            calculated_reaction_rates[reaction_name] = reaction_rate
            for product_key in products:
                if product_key in self.substrates:
                    product_idx = self.substrates[product_key]
                    if product_idx < x.shape[2]:
                        output[:, :, product_idx] += torch.clamp(reaction_rate, min=0.0)
            for substrate_key in substrates:
                if substrate_key in self.substrates:
                    s_idx = self.substrates[substrate_key]
                    if s_idx < x.shape[2]:
                        output[:, :, s_idx] -= torch.clamp(reaction_rate, min=0.0)
        transformed_x = self.transform(x)
        combined_output = transformed_x + output
        return combined_output, calculated_reaction_rates

class SewerGNN(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int = None, dropout_rate: float = 0.5, num_nodes: int = None):
        super(SewerGNN, self).__init__()
        self.dropout_rate = dropout_rate
        self.output_dim = output_dim if output_dim is not None else input_dim
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.bn3 = nn.BatchNorm1d(hidden_dim)
        self.time_attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=1, dropout=dropout_rate)
        self.time_norm = nn.LayerNorm(hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, self.output_dim)
        self.monod_layer = MonodKineticsLayer(hidden_dim, num_nodes)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.norm3 = nn.LayerNorm(hidden_dim)
        self.residual1 = nn.Linear(input_dim, hidden_dim)
        self.residual2 = nn.Linear(hidden_dim, hidden_dim)
        self.prediction_enhancer = nn.Sequential(
            nn.Linear(hidden_dim + input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, self.output_dim)
        )
    def forward(self, x, edge_index) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        num_nodes, time_steps, features = x.shape
        gcn_outputs = []
        all_reaction_rates_over_gnn_time_steps = []
        for t in range(time_steps):
            x_t = x[:, t, :]
            residual1 = self.residual1(x_t)
            h = self.conv1(x_t, edge_index)
            h = self.bn1(h)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout_rate, training=self.training)
            h_monod_input = h.unsqueeze(1)
            h_transformed_by_monod, current_ts_reaction_rates = self.monod_layer(h_monod_input)
            h = h_transformed_by_monod.squeeze(1)
            all_reaction_rates_over_gnn_time_steps.append(current_ts_reaction_rates)
            h = self.conv2(h, edge_index)
            h = self.bn2(h)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout_rate, training=self.training)
            h = self.conv3(h, edge_index)
            h = self.bn3(h)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout_rate, training=self.training)
            residual2 = self.residual2(h)
            h = h + residual2
            gcn_outputs.append(h)
        stacked_output = torch.stack(gcn_outputs, dim=1)
        stacked_output_transposed = stacked_output.transpose(0, 1)
        attn_output, _ = self.time_attention(
            stacked_output_transposed, stacked_output_transposed, stacked_output_transposed
        )
        attn_output = attn_output + stacked_output_transposed
        attn_output = self.time_norm(attn_output)
        attn_output_processed = attn_output.transpose(0, 1)
        enhanced_features_list = []
        for t_idx in range(time_steps):
            concat_features_at_t = torch.cat([attn_output_processed[:, t_idx, :], x[:, t_idx, :]], dim=1)
            enhanced_pred_at_t = self.prediction_enhancer(concat_features_at_t)
            enhanced_features_list.append(enhanced_pred_at_t)
        final_prediction_output = torch.stack(enhanced_features_list, dim=1)
        final_reaction_rates_for_loss = {}
        if all_reaction_rates_over_gnn_time_steps:
            reaction_names_in_model = all_reaction_rates_over_gnn_time_steps[0].keys()
            for r_name in reaction_names_in_model:
                rates_one_reaction_over_time = [rates_dict_for_timestep[r_name] for rates_dict_for_timestep in all_reaction_rates_over_gnn_time_steps]
                final_reaction_rates_for_loss[r_name] = torch.stack(rates_one_reaction_over_time, dim=1).squeeze(-1)
        return final_prediction_output, final_reaction_rates_for_loss
