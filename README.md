# liveInterferometer

A simple Dash app to show the basic ideas of radio interferometry, using your webcam and OpenCV image detection.

![Web app screenshot](https://github.com/telegraphic/liveInterferometer/blob/master/images/interferometer_demo.jpg?raw=true)

### Credits

This app is based on [liveInterferometer](https://github.com/jack-h/liveInterferometer) by Jack Hickish.

The radio image used as an example is from the [GaLactic and Extragalactic All-sky Murchison Widefield Array](https://www.icrar.org/gleam/) (GLEAM) survey.

### Requirements

You will need a Python installation with the following packages:

* opencv-python
* numpy
* dash
* dash-bootstrap-components
* plotly

You'll also need a webcam.

### Running the app

Download this repo, then from the command line run:

```
> python interferometer_sim.py
```

This will start a web server on http://127.0.0.1:8050. Open up a web browser and type in that address.