def test_bundle_contains_everything_the_api_needs(bundle):
    for key in ("pipeline", "features", "defaults", "brand_defaults", "categories", "metrics"):
        assert key in bundle


def test_model_explains_most_of_the_price_variance(bundle):
    assert bundle["metrics"]["r2"] > 0.8


def test_brand_defaults_reflect_the_brand(bundle):
    bd = bundle["brand_defaults"]
    assert bd["BMW"]["Power"] > bd["Maruti"]["Power"]
