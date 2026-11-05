import copy
import unittest
import torch
from torch import nn
from models.visual_world_model import VWorldModel
from models.effect_world_model import EffectWorldModel, effect_distances, make_idm_head


class TinyEncoder(nn.Module):
    name = "test_encoder"
    emb_dim = 8
    latent_ndim = 1
    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(3*8*8, 8)
    def forward(self, x): return self.projection(x.flatten(1))[:, None]


def make(mode="none", lam=0., cls=EffectWorldModel):
    torch.manual_seed(23)
    kw = dict(image_size=8, num_hist=3, num_pred=1, encoder=TinyEncoder(),
        proprio_encoder=nn.Linear(4,3), action_encoder=nn.Linear(10,2),
        decoder=None, predictor=nn.Linear(13,13), proprio_dim=3, action_dim=2,
        concat_dim=1, num_proprio_repeat=1, num_action_repeat=1,
        train_predictor=True, train_decoder=False, straighten="cos1e-2")
    if cls is EffectWorldModel: kw.update(effect_mode=mode,effect_lambda=lam,idm_action_dim=10)
    return cls(**kw)


def batch():
    torch.manual_seed(9)
    return {"visual":torch.randn(2,4,3,8,8),"proprio":torch.randn(2,4,4),
            "effect_labels":torch.ones(2,4,2,dtype=torch.bool)}, torch.randn(2,4,10)


class EffectTrainingTests(unittest.TestCase):
    def test_b0_and_zero_lambda_preserve_original(self):
        obs, act = batch()
        base = make(cls=VWorldModel)
        y = base(obs,act)
        for mode in ("none","effect","idm"):
            wm = make(mode)
            z = wm(obs,act)
            torch.testing.assert_close(y[0],z[0],rtol=0,atol=0)
            torch.testing.assert_close(y[3],z[3],rtol=0,atol=0)

    def test_unknown_has_no_extra_loss(self):
        obs, act = batch(); obs["effect_labels"].zero_()
        wm = make("effect",lam=2)
        out=wm(obs,act)
        self.assertEqual(float(out[4]["aux_loss"]),0)
        self.assertIsNone(wm._last_effect_z)

    def test_action_token_and_predictor_cannot_solve_auxiliary(self):
        obs,act=batch()
        for mode in ("effect","idm"):
            wm=make(mode,lam=1)
            wm(obs,act)[4]["aux_loss"].backward()
            self.assertTrue(all(p.grad is None or torch.count_nonzero(p.grad) == 0 for p in wm.action_encoder.parameters()))
            self.assertTrue(all(p.grad is None for p in wm.predictor.parameters()))
            self.assertTrue(any(p.grad is not None for p in wm.encoder.parameters()))

    def test_idm_head_does_not_advance_rng(self):
        state=torch.get_rng_state().clone()
        make_idm_head(394,10)
        self.assertTrue(torch.equal(state,torch.get_rng_state()))

    def test_identical_observations_have_zero_distance(self):
        v=torch.randn(2,1,1,8).expand(2,4,1,8)
        p=torch.randn(2,1,3).expand(2,4,3)
        torch.testing.assert_close(effect_distances({"visual":v,"proprio":p}),torch.zeros(2,4,2),atol=0,rtol=0)

    def test_pair_weighting_and_action_target_alignment(self):
        obs,act=batch()
        obs["effect_labels"].zero_();obs["effect_labels"][:,1,0]=True
        wm=make("reweight")
        out=wm(obs,act)
        z=wm.encode(obs,act)
        err=(wm.visual_prop(out[0])-wm.visual_prop(z[:,1:].detach())).square().flatten(2).mean(-1)
        weights=torch.tensor([1.,2.,1.]);weights/=weights.mean()
        torch.testing.assert_close(out[4]["z_loss"],(err*weights).mean())
        wm=make("idm")
        # The last action block happens AFTER the last observed pair and must
        # not be an IDM target. Changing it leaves the auxiliary unchanged.
        first=wm(obs,act)[4]["aux_loss"]
        altered=act.clone();altered[:,-1]+=100
        second=wm(obs,altered)[4]["aux_loss"]
        torch.testing.assert_close(first,second,rtol=0,atol=0)


if __name__ == "__main__": unittest.main()
