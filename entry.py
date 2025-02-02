from onnx_modifier.flask_server import (launch_flask_server,
                                        build_desktop_app)

MODE = "LAUNCH_SERVER"
assert MODE in ["LAUNCH_SERVER", "BUILD_EXE"]

if MODE == "LAUNCH_SERVER":
    launch_flask_server()
else:
    build_desktop_app()