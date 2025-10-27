# Heart Shape with Python Turtle

This project uses the **Python turtle graphics library** to draw a vibrant red heart shape. It’s a beginner-friendly script that visually demonstrates basic movement and drawing commands in Python.

## Features

- Draws a red heart shape using turtle graphics.
- Customizes background color and pen properties for a visually appealing effect.
- Full code included in `heart.py`.

## Getting Started

### Prerequisites

Make sure you have Python installed (version 3.x recommended).

### Install turtle

*turtle* comes pre-installed with standard Python distributions. If you're using a limited environment, you might need to install it via pip (though usually unnecessary):

```bash
pip install PythonTurtle
```

### How to Run

1. Clone this repository or download the `heart.py` file.
2. Open a terminal and navigate to the project directory.
3. Run the following command:

```bash
python heart.py
```

A window will pop up displaying the heart shape drawn by the turtle!

## Code Explanation

- **Background and Pen Settings:**
  - Sets the background to black, the pen size to 2, and turtle speed to 5 for smooth drawing.
- **Heart Drawing:**
  - Uses a helper function to create the curved top of the heart on each side.
  - Fills the heart with red color.
  - Hides the turtle at the end for a clean display.