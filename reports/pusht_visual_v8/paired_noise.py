"""Fresh, reproducible salt/pepper noise for ECP-on/off evaluation pairs."""
import hashlib


def noise_seed(episode_ids, call_index):
    key = ",".join(str(int(x)) for x in episode_ids)
    payload = f"ecp_snp_v1|{key}|{int(call_index)}".encode()
    return int.from_bytes(hashlib.blake2s(payload, digest_size=4).digest(), "little")


def install_pairing(evaluator_class, corrupt_obs_dict):
    original = evaluator_class._corrupt

    def paired_corrupt(self, obs, seed_offset=0, random_per_call=False):
        if not random_per_call or not str(self.ood_corruption).startswith("snp"):
            return original(self, obs, seed_offset, random_per_call)
        try:
            episode_ids = tuple(int(x) for x in self.seed)
        except TypeError:
            episode_ids = (int(self.seed),)
        counters = getattr(self, "_ecp_noise_call_counts", None)
        if counters is None:
            counters = self._ecp_noise_call_counts = {}
        index = counters.get(episode_ids, 0)
        counters[episode_ids] = index + 1
        return corrupt_obs_dict(obs, self.ood_corruption, self.ood_level,
                                seed=noise_seed(episode_ids, index))

    evaluator_class._corrupt = paired_corrupt

