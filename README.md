# launchpad_pro_mk3
Python3 tool for interacting with Novation's Launchpad Pro Mk3 control surface. 

This repository contains two Python modules; the first, for manipulating pitch-class-sets (`harmony.py`)
and the second (`launch.py`) for interacting with a Launchpad Pro Mk3 connected to your computer via USB.
The Launchpad needs to be switched to "programmer mode" (hold "Setup" and press the lowest button on the right, "Print to Clip")
to behave properly; invoking `launch.py` from the terminal will

The script uses the Python package `mido` to interact with the Launchpad and with other midi ports on your system.
That package can be installed with the command `pip install mido`. 