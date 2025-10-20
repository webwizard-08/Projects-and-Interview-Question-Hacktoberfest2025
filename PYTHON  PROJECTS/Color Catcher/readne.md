# Color Catcher

**Author:** Pranshu
**Language:** Python (turtle)  

## About the game
Color Catcher is a fast-paced reflex game where you control a bucket to catch falling colored balls. Each color gives different points, and the game speeds up as your score increases.

## Features
- Simple graphics using Python's built-in `turtle` module  
- Randomly colored falling balls  
- Scoring system with different color values  
- Dynamic difficulty — the more you catch, the faster it gets  
- Clean, object-free interface

## How to play
Run the game:

```bash
python color_catcher.py
```

Use the Left and Right Arrow keys to move the bucket and catch as many balls as possible.

### Scoring
| Color  | Points |
|--------|--------|
| Red    | +5     |
| Blue   | +10    |
| Green  | +15    |
| Yellow | +20    |
| Purple | +25    |

Missing a ball: −2 points.

The game speed increases over time — try to last as long as possible.

## Requirements
- Python 3.7 or above  
- No external libraries required (uses built-in `turtle` and `random`)
