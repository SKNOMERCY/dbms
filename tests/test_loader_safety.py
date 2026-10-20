import pytest

import load_data


def test_loader_requires_an_explicit_reset_opt_in():
    with pytest.raises(SystemExit) as error:
        load_data.main([])

    assert error.value.code == 2
