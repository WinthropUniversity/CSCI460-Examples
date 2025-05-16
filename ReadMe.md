# Example Source Code for Winthrop's CSCI460, Machine Learning Course

This public repo contains examples we're using in our Machine Learning class.  To get them to work, you'll need to add the examples directory to your *PYTHONPATH*:

```
export PYTHONPATH=/path/to/CSCI460X-Examples:${PYTHONPATH}
```

If you are running our Docker container, you can create a file in your *persistent-homedir* called *.bashrc* with the following line in it:

```
export PYTHONPATH=/home/student/CSCI460X-Examples:${PYTHONPATH}
```

## If you need to run on Hopper

To use TensorFlow on Hopper, you will need to create a *virtual environment* and install TF into that.  Then, when you run, you'll need to *activate* that virtual environment.  To install in the first place:

```
# Create a virtual environment
python3 -m venv tensorflow

# Activate that env
source tensorflow/bin/activate

# Install tensorflow
pip3 install tensorflow
```

You only need to do that once.  After that, whenever you want to run, you'll need to put yourself in that virtual environment:
```
source tensorflow/bin/activate
```

You might *also* need to add the examples directory to your `PYTHONPATH`:
```
# Modify the path so that it is correct for your setup
export PYTHONPATH=/home/acc.username/CSCI460-Examples
```
