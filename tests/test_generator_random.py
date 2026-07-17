import h5py
import numpy as np
import pytest

import csdl_alpha as csdl
from csdml import Generator


@pytest.mark.parametrize("backend", ("inline", "jax"))
def test_generator_uses_distinct_reproducible_random_values_per_sample(
    tmp_path, backend
):
    def generate(path):
        recorder = csdl.Recorder(inline=True)
        recorder.start()
        generator = Generator(recorder)

        x = csdl.Variable(name="x", value=np.zeros(1))
        generator.add_input(x, lower=np.zeros(1), upper=np.ones(1))
        random_values = csdl.normal((4, 2)) + 0.0 * x
        random_values.add_name("random_values")
        generator.add_output(random_values)

        generator.generate(
            filename=str(path),
            n_samples=3,
            seed=11,
            random_seed=29,
            backend=backend,
            device="cpu",
            save_inputs=False,
        )
        recorder.stop()

        with h5py.File(path, "r") as stream:
            assert all(list(stream[f"sample_{i}"]) == ["random_values"] for i in range(3))
            return [stream[f"sample_{i}"]["random_values"][...] for i in range(3)]

    first = generate(tmp_path / f"first_{backend}.hdf5")
    second = generate(tmp_path / f"second_{backend}.hdf5")

    assert not np.array_equal(first[0], first[1])
    for first_sample, second_sample in zip(first, second):
        np.testing.assert_array_equal(first_sample, second_sample)
