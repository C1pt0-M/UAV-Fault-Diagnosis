from __future__ import annotations

from uav_dashboard_api import create_app
from uav_dashboard_api.__main__ import main


app = create_app()


if __name__ == "__main__":
    main()
