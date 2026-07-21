DASHBOARD_ENDPOINTS = {
    ("/api/overview", "GET", "dashboard.get_overview"),
    ("/api/models", "GET", "dashboard.get_models"),
    ("/api/assets/upload", "POST", "dashboard.upload_asset"),
    ("/api/assets", "GET", "dashboard.list_assets"),
    ("/api/assets/<int:asset_id>", "GET", "dashboard.get_asset"),
    ("/api/assets/<int:asset_id>/download", "GET", "dashboard.download_asset"),
    ("/api/assets/<int:asset_id>", "DELETE", "dashboard.delete_asset"),
    ("/api/assets/<int:asset_id>/prepare", "POST", "dashboard.prepare_asset"),
    ("/api/runs", "POST", "dashboard.create_run"),
    ("/api/runs", "GET", "dashboard.list_runs"),
    ("/api/runs/<int:run_id>", "GET", "dashboard.get_run"),
    ("/api/runs/<int:run_id>/result", "GET", "dashboard.get_run_result"),
    ("/api/analysis/fields", "GET", "dashboard.get_analysis_fields"),
    ("/api/analysis/timeseries", "GET", "dashboard.get_analysis_timeseries"),
    ("/api/analysis/fault_prob_curve", "GET", "dashboard.get_fault_prob_curve"),
}


def test_dashboard_endpoint_inventory_is_stable(app):
    actual = {
        (rule.rule, method, rule.endpoint)
        for rule in app.url_map.iter_rules()
        if rule.endpoint.startswith("dashboard.")
        for method in rule.methods - {"HEAD", "OPTIONS"}
    }

    assert actual == DASHBOARD_ENDPOINTS


def test_dashboard_endpoints_require_authentication(app, client):
    for rule, method, _endpoint in DASHBOARD_ENDPOINTS:
        url = rule.replace("<int:asset_id>", "1").replace("<int:run_id>", "1")
        response = client.open(url, method=method)
        assert response.status_code == 401, (method, url, response.get_json())


def test_health_route_has_one_source_of_truth(app):
    rules = [rule for rule in app.url_map.iter_rules() if rule.rule == "/health"]

    assert [(rule.rule, rule.endpoint) for rule in rules] == [
        ("/health", "health.health_check")
    ]
