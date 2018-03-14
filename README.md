# How to use Keras in a web server

This project is a minimal exemple on how to use a Keras model in a web server.
I hope this helps!

*NOTE* Since it's just an example, gpu are not enabled. Delete the line [here](https://github.com/Dref360/tuto_keras_web/blob/master/keras_model.py#L3) to activate GPUs.


## Requirements
* Keras
* Flask
* Opencv3

## How to run
* Start the server `python web.py`
* Send a request (I recommend Postman)
  * POST, http://127.0.0.1:5000/hello
  * The body should have field `'file'` with an image file.
* You should get a response with an imagenet class.
