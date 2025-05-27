import time
import sys
import gi

gi.require_version("Gst", "1.0")
from gi.repository import Gst

def capture_images():
    serial_1 = "32910328"
    serial_2 = "32910323"

    pipeline_1 = Gst.parse_launch("tcambin name=bin1"
                                  " ! video/x-raw,format=BGRx,width=1920,height=1080,framerate=30/1"
                                  " ! videoconvert"
                                  " ! jpegenc"
                                  " ! filesink name=fsink1")

    pipeline_2 = Gst.parse_launch("tcambin name=bin2"
                                  " ! video/x-raw,format=BGRx,width=1920,height=1080,framerate=30/1"
                                  " ! videoconvert"
                                  " ! jpegenc"
                                  " ! filesink name=fsink2")

    if serial_1 is not None:
        camera_1 = pipeline_1.get_by_name("bin1")
        camera_1.set_property("serial", serial_1)

    if serial_2 is not None:
        camera_2 = pipeline_2.get_by_name("bin2")
        camera_2.set_property("serial", serial_2)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    #On définit le path pour enregistrer les images
    file_location_1 = f"camera1_3D/camera_1_image_{timestamp}.jpg"
    file_location_2 = f"camera2_3D/camera_2_image_{timestamp}.jpg"

    fsink_1 = pipeline_1.get_by_name("fsink1")
    fsink_1.set_property("location", file_location_1)

    fsink_2 = pipeline_2.get_by_name("fsink2")
    fsink_2.set_property("location", file_location_2)

    pipeline_1.set_state(Gst.State.PLAYING)
    pipeline_2.set_state(Gst.State.PLAYING)

    print(f"Capturing images... ({file_location_1}, {file_location_2})")
    time.sleep(2)
    pipeline_1.set_state(Gst.State.NULL)
    pipeline_2.set_state(Gst.State.NULL)
    print("Images saved!")

def main():
    Gst.init(sys.argv)
    Gst.debug_set_default_threshold(Gst.DebugLevel.WARNING)
    print("Tape 't' puis Entrée pour prendre des photos avec les deux caméras. Ctrl-C pour quitter.")

    try:
        while True:
            key = input("Appuie sur 't' puis Entrée pour prendre une photo (ou Ctrl-C pour quitter) : ").strip().lower()
            if key == 't':
                capture_images()
    except KeyboardInterrupt:
        print("\nArrêt du programme.")

if __name__ == "__main__":
    main()
