from flask import Blueprint

from uav_dashboard_api.routes import (
    dashboard_analysis,
    dashboard_assets,
    dashboard_overview,
    dashboard_runs,
)


bp = Blueprint("dashboard", __name__)

dashboard_overview.register_routes(bp)
dashboard_assets.register_routes(bp)
dashboard_runs.register_routes(bp)
dashboard_analysis.register_routes(bp)
