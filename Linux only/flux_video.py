import gi
import sys
import time

gi.require_version("Gst", "1.0")
from gi.repository import Gst

def main():
    Gst.init(sys.argv)

    serial = "32910328"  #numéro de série de ta caméra
    #serial = "32910323"

    pipeline = Gst.parse_launch(
        f"tcambin serial={serial} ! videoconvert ! autovideosink sync=false"
    )

    pipeline.set_state(Gst.State.PLAYING)
    print("Ctrl-C pour quitter")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        pipeline.set_state(Gst.State.NULL)

if __name__ == "__main__":
    main()
