import sys
import gi

gi.require_version("Tcam", "1.0")
gi.require_version("Gst", "1.0")
gi.require_version("GLib", "2.0")

from gi.repository import Tcam, Gst, GLib


def print_properties(camera):
    """
    Print selected properties
    """
    try:

        property_exposure_auto = camera.get_tcam_property("ExposureAuto")

        print(property_exposure_auto.get_value())

        value = camera.get_tcam_enumeration("ExposureAuto")

        print(f"Exposure Auto has value: {value}")

        value = camera.get_tcam_enumeration("GainAuto")

        print("Gain Auto has value: {}".format(value))

        value = camera.get_tcam_float("ExposureTime")

        print("ExposureTimer has value: {}".format(value))

    except GLib.Error as err:

        print(f"{err.message}")


def block_until_playing(pipeline):

    while True:
        # wait 0.1 seconds for something to happen
        change_return, state, pending = pipeline.get_state(100000000)
        if change_return == Gst.StateChangeReturn.SUCCESS:
            return True
        elif change_return == Gst.StateChangeReturn.FAILURE:
            print("Failed to change state {} {} {}".format(change_return,
                                                           state,
                                                           pending))
            return False


def main():

    Gst.init(sys.argv)
    Gst.debug_set_default_threshold(Gst.DebugLevel.WARNING)

    # Set this to a serial string for a specific camera
    serial = "32910328"

    camera = Gst.ElementFactory.make("tcambin")

    if serial:
        # This is gstreamer set_property
        camera.set_property("serial", serial)

    # in the READY state the camera will always be initialized
    camera.set_state(Gst.State.READY)


    # Set properties
    try:
        camera.set_tcam_enumeration("ExposureAuto", "Off")
        camera.set_tcam_enumeration("GainAuto", "Off")

        camera.set_tcam_float("ExposureTime", 20000)
        print("Properties Set :")

    except GLib.Error as err:
        # if setting properties fail, print the reason
        print(f"error : {err.message}")

    print_properties(camera)

    # cleanup, reset state
    camera.set_state(Gst.State.NULL)


if __name__ == "__main__":
    sys.exit(main())
