def pytest_configure(config):
    config.addinivalue_line("markers", "e2e: Playwright UI tests against a live stack")
