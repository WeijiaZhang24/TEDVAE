import logging

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import pyro
import pyro.distributions as dist
from pyro import poutine
from pyro.infer import SVI, Trace_ELBO
from pyro.infer.util import torch_item
from pyro.nn import PyroModule
from pyro.optim import ClippedAdam
from pyro.util import torch_isnan

logger = logging.getLogger(__name__)


class FullyConnected(nn.Sequential):
    """
    Fully connected multi-layer network with ELU activations.
    """
    def __init__(self, sizes, final_activation=None):
        layers = []
        for in_size, out_size in zip(sizes, sizes[1:]):
            layers.append(nn.Linear(in_size, out_size))
            layers.append(nn.ELU())
        layers.pop(-1)
        if final_activation is not None:
            layers.append(final_activation)
        super().__init__(*layers)

    def append(self, layer):
        assert isinstance(layer, nn.Module)
        self.add_module(str(len(self)), layer)


class DistributionNet(nn.Module):
    """
    Base class for distribution nets.
    """
    @staticmethod
    def get_class(dtype):
        """
        Get a subclass by a prefix of its name, e.g.::

            assert DistributionNet.get_class("bernoulli") is BernoulliNet
        """
        for cls in DistributionNet.__subclasses__():
            if cls.__name__.lower() == dtype + "net":
                return cls
        raise ValueError("dtype not supported: {}".format(dtype))


class BernoulliNet(DistributionNet):
    """
    :class:`FullyConnected` network outputting a single ``logits`` value.

    This is used to represent a conditional probability distribution of a
    single Bernoulli random variable conditioned on a ``sizes[0]``-sized real
    value, for example::

        net = BernoulliNet([3, 4])
        z = torch.randn(3)
        logits, = net(z)
        t = net.make_dist(logits).sample()
    """
    def __init__(self, sizes):
        assert len(sizes) >= 1
        super().__init__()
        self.fc = FullyConnected(sizes + [1])

    def forward(self, x):
        logits = self.fc(x).squeeze(-1).clamp(min=-10, max=10)
        return logits,

    @staticmethod
    def make_dist(logits):
        return dist.Bernoulli(logits=logits)


class ExponentialNet(DistributionNet):
    """
    :class:`FullyConnected` network outputting a constrained ``rate``.

    This is used to represent a conditional probability distribution of a
    single Normal random variable conditioned on a ``sizes[0]``-size real
    value, for example::

        net = ExponentialNet([3, 4])
        x = torch.randn(3)
        rate, = net(x)
        y = net.make_dist(rate).sample()
    """
    def __init__(self, sizes):
        assert len(sizes) >= 1
        super().__init__()
        self.fc = FullyConnected(sizes + [1])

    def forward(self, x):
        scale = nn.functional.softplus(self.fc(x).squeeze(-1)).clamp(min=1e-3, max=1e6)
        rate = scale.reciprocal()
        return rate,

    @staticmethod
    def make_dist(rate):
        return dist.Exponential(rate)


class LaplaceNet(DistributionNet):
    """
    :class:`FullyConnected` network outputting a constrained ``loc,scale``
    pair.

    This is used to represent a conditional probability distribution of a
    single Laplace random variable conditioned on a ``sizes[0]``-size real
    value, for example::

        net = LaplaceNet([3, 4])
        x = torch.randn(3)
        loc, scale = net(x)
        y = net.make_dist(loc, scale).sample()
    """
    def __init__(self, sizes):
        assert len(sizes) >= 1
        super().__init__()
        self.fc = FullyConnected(sizes + [2])

    def forward(self, x):
        loc_scale = self.fc(x)
        loc = loc_scale[..., 0].clamp(min=-1e6, max=1e6)
        scale = nn.functional.softplus(loc_scale[..., 1]).clamp(min=1e-3, max=1e6)
        return loc, scale

    @staticmethod
    def make_dist(loc, scale):
        return dist.Laplace(loc, scale)


class NormalNet(DistributionNet):
    """
    :class:`FullyConnected` network outputting a constrained ``loc,scale``
    pair.

    This is used to represent a conditional probability distribution of a
    single Normal random variable conditioned on a ``sizes[0]``-size real
    value, for example::

        net = NormalNet([3, 4])
        x = torch.randn(3)
        loc, scale = net(x)
        y = net.make_dist(loc, scale).sample()
    """
    def __init__(self, sizes):
        assert len(sizes) >= 1
        super().__init__()
        self.fc = FullyConnected(sizes + [2])

    def forward(self, x):
        loc_scale = self.fc(x)
        loc = loc_scale[..., 0].clamp(min=-1e6, max=1e6)
        scale = nn.functional.softplus(loc_scale[..., 1]).clamp(min=1e-3, max=1e6)
        return loc, scale

    @staticmethod
    def make_dist(loc, scale):
        return dist.Normal(loc, scale)


class StudentTNet(DistributionNet):
    """
    :class:`FullyConnected` network outputting a constrained ``df,loc,scale``
    triple, with shared ``df > 1``.

    This is used to represent a conditional probability distribution of a
    single Student's t random variable conditioned on a ``sizes[0]``-size real
    value, for example::

        net = StudentTNet([3, 4])
        x = torch.randn(3)
        df, loc, scale = net(x)
        y = net.make_dist(df, loc, scale).sample()
    """
    def __init__(self, sizes):
        assert len(sizes) >= 1
        super().__init__()
        self.fc = FullyConnected(sizes + [2])
        self.df_unconstrained = nn.Parameter(torch.tensor(0.0))

    def forward(self, x):
        loc_scale = self.fc(x)
        loc = loc_scale[..., 0].clamp(min=-1e6, max=1e6)
        scale = nn.functional.softplus(loc_scale[..., 1]).clamp(min=1e-3, max=1e6)
        df = nn.functional.softplus(self.df_unconstrained).add(1).expand_as(loc)
        return df, loc, scale

    @staticmethod
    def make_dist(df, loc, scale):
        return dist.StudentT(df, loc, scale)


class DiagNormalNet(nn.Module):
    """
    :class:`FullyConnected` network outputting a constrained ``loc,scale``
    pair.

    This is used to represent a conditional probability distribution of a
    ``sizes[-1]``-sized diagonal Normal random variable conditioned on a
    ``sizes[0]``-size real value, for example::

        net = DiagNormalNet([3, 4, 5])
        z = torch.randn(3)
        loc, scale = net(z)
        x = dist.Normal(loc, scale).sample()

    This is intended for the latent ``z`` distribution and the prewhitened
    ``x`` features, and conservatively clips ``loc`` and ``scale`` values.
    """
    def __init__(self, sizes):
        assert len(sizes) >= 2
        self.dim = sizes[-1]
        super().__init__()
        self.fc = FullyConnected(sizes[:-1] + [self.dim * 2])

    def forward(self, x):
        loc_scale = self.fc(x)
        loc = loc_scale[..., :self.dim].clamp(min=-1e2, max=1e2)
        scale = nn.functional.softplus(loc_scale[..., self.dim:]).add(1e-3).clamp(max=1e2)
        return loc, scale

class DiagBernoulliNet(nn.Module):
    """
    :class:`FullyConnected` network outputting a single ``logits`` value.

    This is used to represent a conditional probability distribution of a
    single Bernoulli random variable conditioned on a ``sizes[0]``-sized real
    value, for example::

        net = DiagBernoulliNet([3, 4, 5])
        z = torch.randn(3)
        logits, = net(z)
        t = net.make_dist(logits).sample()
    """
    def __init__(self, sizes):
        assert len(sizes) >= 2
        self.dim = sizes[-1]
        super().__init__()
        self.fc = FullyConnected(sizes[:-1] + [self.dim])

    def forward(self, x):
        logits = self.fc(x).squeeze(-1).clamp(min=0, max=11)
        return logits

    @staticmethod
    def make_dist(logits):
        return dist.Bernoulli(logits=logits)


class PreWhitener(nn.Module):
    """
    Data pre-whitener.
    """
    def __init__(self, data):
        super().__init__()
        with torch.no_grad():
            loc = data.mean(0)
            scale = data.std(0)
            scale[~(scale > 0)] = 1.
            self.register_buffer("loc", loc)
            self.register_buffer("inv_scale", scale.reciprocal())

    def forward(self, data):
        return (data - self.loc) * self.inv_scale



class Guide(PyroModule):
    """
    Inference model for causal effect estimation with latent confounder ``z``
    and binary treatment ``t``::
        z ~ p(z|x)  # latent confounder, an embedding
        zt ~ p(zt|x)
        zy ~ p(zy|x)
        t ~ p(t|z,zt) # treatment
        y ~ p(y|t,z,zy)    # outcome

    Each of these distributions is defined by a neural network.  The ``y`` and
    ``z`` distributions are defined by disjoint pairs of neural networks
    defining ``p(-|t=0,...)`` and ``p(-|t=1,...)``; this allows highly
    imbalanced treatment.

    :param dict config: A dict specifying ``feature_dim``, ``latent_dim``,
        ``hidden_dim``, ``num_layers``, and ``outcome_dist``.
    """
    def __init__(self, config):
        self.latent_dim = config["latent_dim"]
        self.latent_dim_t = config["latent_dim_t"]
        self.latent_dim_y = config["latent_dim_y"]
        
        OutcomeNet = DistributionNet.get_class(config["outcome_dist"])
        super().__init__()
        # self.t_nn = BernoulliNet([config["feature_dim"]])
        self.t_nn = BernoulliNet([config["latent_dim"]+config["latent_dim_t"]])

        # The y and z networks both follow an architecture where the first few
        # layers are shared for t in {0,1}, but the final layer is split
        # between the two t values.
        self.y_nn = FullyConnected([config["latent_dim"] + config["latent_dim_y"]] +
                                   [config["hidden_dim"]] * (config["num_layers"] - 1),
                                   final_activation=nn.ELU())
        self.y0_nn = OutcomeNet([config["hidden_dim"]])
        self.y1_nn = OutcomeNet([config["hidden_dim"]])
        
        self.z_nn = FullyConnected([config["feature_dim"]] +
                                   [config["hidden_dim"]] * (config["num_layers"] - 1),
                                   final_activation=nn.ELU())
        

        self.z_out_nn = DiagNormalNet([config["hidden_dim"], config["latent_dim"]])
        
        
        self.zt_nn = FullyConnected([config["feature_dim"]] +
                                   [config["hidden_dim"]] * (config["num_layers"] - 1),
                                   final_activation=nn.ELU())
        
        self.zt_out_nn = DiagNormalNet([config["hidden_dim"], config["latent_dim_t"]])
        
        
        self.zy_nn = FullyConnected([config["feature_dim"]] +
                                   [config["hidden_dim"]] * (config["num_layers"] - 1),
                                   final_activation=nn.ELU())
        self.zy_out_nn = DiagNormalNet([config["hidden_dim"], config["latent_dim_y"]])

    def forward(self, x, t=None, y=None, size=None):
        if size is None:
            size = x.size(0)
        with pyro.plate("data", size, subsample=x):
            # The t and y sites are needed for prediction, and participate in
            # the auxiliary CEVAE loss. We mark them auxiliary to indicate they
            # do not correspond to latent variables during training.
            z=pyro.sample("z", self.z_dist(x))
            zt=pyro.sample("zt", self.zt_dist(x))
            zy=pyro.sample("zy", self.zy_dist(x))
            
            t = pyro.sample("t", self.t_dist(z,zt), obs=t, infer={"is_auxiliary": True})

            y = pyro.sample("y", self.y_dist(t,z,zy), obs=y, infer={"is_auxiliary": True})
            # The z site participates only in the usual ELBO loss.

    def z_mean(self, x, t=None):
        with pyro.plate("data", x.size(0)):
            z = pyro.sample("z", self.z_dist(x))
            zt = pyro.sample("zt", self.zt_dist(x))
            zy = pyro.sample("zy", self.zy_dist(x))
        return z,zt,zy

    def t_dist(self, z, zt):
        input_concat = torch.cat((z,zt),-1)
        logits, = self.t_nn(input_concat)
        return dist.Bernoulli(logits=logits)

    def y_dist(self, t, z, zy):
        # The first n-1 layers are identical for all t values.
        x = torch.cat((z,zy),-1)
        hidden = self.y_nn(x)
        # In the final layer params are not shared among t values.
        params0 = self.y0_nn(hidden)
        params1 = self.y1_nn(hidden)
        t = t.bool()
        params = [torch.where(t, p1, p0) for p0, p1 in zip(params0, params1)]
        return self.y0_nn.make_dist(*params)

    def z_dist(self, x):
        # hidden = self.z_nn(x)
        hidden = self.z_nn(x.float())
        params = self.z_out_nn(hidden)
        return dist.Normal(*params).to_event(1)

    def zt_dist(self, x):
        hidden = self.zt_nn(x.float())
        params = self.zt_out_nn(hidden)
        return dist.Normal(*params).to_event(1)
    
    def zy_dist(self, x):
        hidden = self.zy_nn(x.float())
        params = self.zy_out_nn(hidden)
        return dist.Normal(*params).to_event(1)


class TraceCausalEffect_ELBO(Trace_ELBO):
    """
    Loss function for training a :class:`TEDVAE`.
    From [1], the CEVAE objective (to maximize) is::

        -loss = ELBO + log q(t|z,zt) + log q(y|t,z,zy)
    """
    def _differentiable_loss_particle(self, model_trace, guide_trace):
        # Construct -ELBO part.
        blocked_names = [name for name, site in guide_trace.nodes.items()
                         if site["type"] == "sample" and site["is_observed"]]
        blocked_guide_trace = guide_trace.copy()
        for name in blocked_names:
            del blocked_guide_trace.nodes[name]
        loss, surrogate_loss = super()._differentiable_loss_particle(
            model_trace, blocked_guide_trace)

        # Add log q terms.
        for name in blocked_names:
            log_q = guide_trace.nodes[name]["log_prob_sum"]
            loss = loss - 100* torch_item(log_q)
            surrogate_loss = surrogate_loss - 100* log_q

        return loss, surrogate_loss

    @torch.no_grad()
    def loss(self, model, guide, *args, **kwargs):
        return torch_item(self.differentiable_loss(model, guide, *args, **kwargs))



class Model(PyroModule):
    """
    Generative model for a causal model with latent confounder ``z`` and binary
    treatment ``t``::

        z ~ p(z)      # latent confounder
        zt ~ p(zt)
        zy ~ p（zy)
        x ~ p(x|z,zt,zy)
        t ~ p(t|z,zt)
        y ~ p(y|t,z,zy)

    Each of these distributions is defined by a neural network.  The ``y``
    distribution is defined by a disjoint pair of neural networks defining
    ``p(y|t=0,z,zy)`` and ``p(y|t=1,z,zy)``; this allows highly imbalanced treatment.

    :param dict config: A dict specifying ``feature_dim``, ``latent_dim``,
        ``hidden_dim``, ``num_layers``, and ``outcome_dist``.
    """
    def __init__(self, config):
        self.latent_dim = config["latent_dim"]
        self.latent_dim_t = config["latent_dim_t"]
        self.latent_dim_y = config["latent_dim_y"]
        self.binfeats = config["binary_dim"]
        self.contfeats = config["continuous_dim"]

        super().__init__()
        self.x_nn = DiagNormalNet([config["latent_dim"]+config["latent_dim_t"]+config["latent_dim_y"]] +
                                  [config["hidden_dim"]] * config["num_layers"] +
                                  [len(config["continuous_dim"])])
        self.x2_nn = DiagBernoulliNet([config["latent_dim"]+config["latent_dim_t"]+config["latent_dim_y"]] +
                                  [config["hidden_dim"]] * config["num_layers"] +
                                  [len(config["binary_dim"])])
        OutcomeNet = DistributionNet.get_class(config["outcome_dist"])
        # The y network is split between the two t values.
        self.y0_nn = OutcomeNet([config["latent_dim"]+config["latent_dim_y"]] +
                                [config["hidden_dim"]] * config["num_layers"])
        self.y1_nn = OutcomeNet([config["latent_dim"]+config["latent_dim_y"]] +
                                [config["hidden_dim"]] * config["num_layers"])
        self.t_nn = BernoulliNet([config["latent_dim"]+config["latent_dim_t"]])

    def forward(self, x, t=None, y=None, size=None):
        if size is None:
            size = x.size(0)
        with pyro.plate("data", size, subsample=x):
            z = pyro.sample("z", self.z_dist())
            zt = pyro.sample("zt", self.zt_dist())
            zy = pyro.sample("zy", self.zy_dist())

            # x = pyro.sample("x", self.x_dist(z, zt, zy), obs=x)
            x_binary = pyro.sample("x_bin", self.x_dist_binary(z, zt, zy), obs=x[:,self.binfeats])
            x_continuous = pyro.sample("x_cont", self.x_dist_continuous(z, zt, zy), obs=x[:,self.contfeats])
            x = torch.cat((x_binary,x_continuous), -1)
            # x = pyro.sample("x", self.x_dist_binary(z, zt, zy), obs=x)
            t = pyro.sample("t", self.t_dist(z, zt), obs=t)
            y = pyro.sample("y", self.y_dist(t, z, zy), obs=y)

        return y

    def y_mean(self, x, t=None):
        with pyro.plate("data", x.size(0)):
            z = pyro.sample("z", self.z_dist())
            zt = pyro.sample("zt", self.zt_dist())
            zy = pyro.sample("zy", self.zy_dist())
            x_binary = pyro.sample("x_bin", self.x_dist_binary(z, zt, zy), obs=x[:,self.binfeats])
            x_continuous = pyro.sample("x_cont", self.x_dist_continuous(z, zt, zy), obs=x[:,self.contfeats])
            x = torch.cat((x_binary,x_continuous), -1)
            t = pyro.sample("t", self.t_dist(z, zt), obs=t)
        return self.y_dist(t, z, zy).mean

    def z_dist(self):
        return dist.Normal(0, 1).expand([self.latent_dim]).to_event(1)
    
    def zt_dist(self):
        return dist.Normal(0, 1).expand([self.latent_dim_t]).to_event(1)

    def zy_dist(self):
        return dist.Normal(0, 1).expand([self.latent_dim_y]).to_event(1)
    
    def x_dist_continuous(self, z, zt, zy):
        z_concat = torch.cat((z,zt,zy), -1)
        loc, scale = self.x_nn(z_concat)
        return dist.Normal(loc, scale).to_event(1)
    
    def x_dist_binary(self, z, zt, zy):
        z_concat = torch.cat((z,zt,zy), -1)
        logits = self.x2_nn(z_concat)
        return dist.Bernoulli(logits=logits).to_event(1)


    def y_dist(self, t, z, zy):
        # Parameters are not shared among t values.
        z_concat = torch.cat((z, zy), -1)
        params0 = self.y0_nn(z_concat)
        params1 = self.y1_nn(z_concat)
        t = t.bool()
        params = [torch.where(t, p1, p0) for p0, p1 in zip(params0, params1)]
        return self.y0_nn.make_dist(*params)

    def t_dist(self, z,zt):
        z_concat = torch.cat((z,zt), -1)
        logits, = self.t_nn(z_concat)
        return dist.Bernoulli(logits=logits)

class TEDVAE(nn.Module):
    def __init__(self, feature_dim, continuous_dim, binary_dim, outcome_dist="normal", 
                 latent_dim=20, latent_dim_t=20, latent_dim_y=20 , hidden_dim=200, num_layers=3, num_samples=100):
        config = dict(feature_dim=feature_dim, latent_dim=latent_dim,
                      latent_dim_t = latent_dim_t, latent_dim_y = latent_dim_y,
                      hidden_dim=hidden_dim, num_layers=num_layers, continuous_dim = continuous_dim, binary_dim = binary_dim,
                      num_samples=num_samples)
        # for name, size in config.items():
        #     if not (isinstance(size, int) and size > 0):
        #         raise ValueError("Expected {} > 0 but got {}".format(name, size))
        config["outcome_dist"] = outcome_dist
        self.feature_dim = feature_dim
        self.num_samples = num_samples

        super().__init__()
        self.model = Model(config)
        self.guide = Guide(config)
        # self.to_dev
        self.cuda()
    def fit(self, x, t, y,
            num_epochs=100,
            batch_size=100,
            learning_rate=1e-3,
            learning_rate_decay=0.1,
            weight_decay=1e-4):
        """
        Train using :class:`~pyro.infer.svi.SVI` with the
        :class:`TraceCausalEffect_ELBO` loss.

        :param ~torch.Tensor x:
        :param ~torch.Tensor t:
        :param ~torch.Tensor y:
        :param int num_epochs: Number of training epochs. Defaults to 100.
        :param int batch_size: Batch size. Defaults to 100.
        :param float learning_rate: Learning rate. Defaults to 1e-3.
        :param float learning_rate_decay: Learning rate decay over all epochs;
            the per-step decay rate will depend on batch size and number of epochs
            such that the initial learning rate will be ``learning_rate`` and the final
            learning rate will be ``learning_rate * learning_rate_decay``.
            Defaults to 0.1.
        :param float weight_decay: Weight decay. Defaults to 1e-4.
        :return: list of epoch losses
        """
        assert x.dim() == 2 and x.size(-1) == self.feature_dim
        assert t.shape == x.shape[:1]
        assert y.shape == y.shape[:1]
        # self.whiten = PreWhitener(x)

        dataset = TensorDataset(x, t, y)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        logger.info("Training with {} minibatches per epoch".format(len(dataloader)))
        num_steps = num_epochs * len(dataloader)
        optim = ClippedAdam({"lr": learning_rate,
                             "weight_decay": weight_decay,
                             "lrd": learning_rate_decay ** (1 / num_steps)})
        svi = SVI(self.model, self.guide, optim, TraceCausalEffect_ELBO())
        losses = []
        for epoch in range(num_epochs):
            for x, t, y in dataloader:
                # x = self.whiten(x)
                loss = svi.step(x, t, y, size=len(dataset)) / len(dataset)
                # print(loss)
                logger.debug("step {: >5d} loss = {:0.6g}".format(len(losses), loss))
                assert not torch_isnan(loss)
                losses.append(loss)
        return losses

    @torch.no_grad()
    def ite(self, x, ym, ys, num_samples=None, batch_size=None):
        r"""
        Computes Individual Treatment Effect for a batch of data ``x``.

        .. math::

            ITE(x) = \mathbb E\bigl[ \mathbf y \mid \mathbf X=x, do(\mathbf t=1) \bigr]
                   - \mathbb E\bigl[ \mathbf y \mid \mathbf X=x, do(\mathbf t=0) \bigr]

        This has complexity ``O(len(x) * num_samples ** 2)``.

        :param ~torch.Tensor x: A batch of data.
        :param int num_samples: The number of monte carlo samples.
            Defaults to ``self.num_samples`` which defaults to ``100``.
        :param int batch_size: Batch size. Defaults to ``len(x)``.
        :return: A ``len(x)``-sized tensor of estimated effects.
        :rtype: ~torch.Tensor
        """
        if num_samples is None:
            num_samples = self.num_samples
        if not torch._C._get_tracing_state():
            assert x.dim() == 2 and x.size(-1) == self.feature_dim

        dataloader = [x] if batch_size is None else DataLoader(x, batch_size=batch_size)
        logger.info("Evaluating {} minibatches".format(len(dataloader)))
        result = []
        for x in dataloader:
            # x = self.whiten(x)
            with pyro.plate("num_particles", num_samples, dim=-2):
                with poutine.trace() as tr, poutine.block(hide=["y", "t"]):
                    self.guide(x)
                with poutine.do(data=dict(t=torch.zeros(()))):
                    y0 = poutine.replay(self.model.y_mean, tr.trace)(x) * ys + ym
                with poutine.do(data=dict(t=torch.ones(()))):
                    y1 = poutine.replay(self.model.y_mean, tr.trace)(x) * ys + ym
            ite = (y1 - y0).mean(0)
            if not torch._C._get_tracing_state():
                logger.debug("batch ate = {:0.6g}".format(ite.mean()))
            result.append(ite)
        return torch.cat(result)

    def to_script_module(self):
        """
        Compile this module using :func:`torch.jit.trace_module` ,
        assuming self has already been fit to data.

        :return: A traced version of self with an :meth:`ite` method.
        :rtype: torch.jit.ScriptModule
        """
        self.train(False)
        fake_x = torch.randn(2, self.feature_dim)
        with pyro.validation_enabled(False):
            # Disable check_trace due to nondeterministic nodes.
            result = torch.jit.trace_module(self, {"ite": (fake_x,)}, check_trace=False)
        return result
