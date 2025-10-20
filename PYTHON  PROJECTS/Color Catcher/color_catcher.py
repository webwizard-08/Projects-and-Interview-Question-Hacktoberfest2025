import turtle
import random
import time

# Screen setup
wn = turtle.Screen()
wn.title("🎨 Color Catcher Game")
wn.bgcolor("black")
wn.setup(width=600, height=600)
wn.tracer(0)

# Player (the bucket)
bucket = turtle.Turtle()
bucket.shape("square")
bucket.color("white")
bucket.shapesize(stretch_wid=1, stretch_len=5)
bucket.penup()
bucket.goto(0, -250)

# Falling ball
ball = turtle.Turtle()
ball.shape("circle")
ball.color("red")
ball.penup()
ball.speed(0)
ball.goto(random.randint(-250, 250), 250)

# Score
score = 0
speed = 0.04
colors = ["red", "blue", "green", "yellow", "purple"]

# Scoreboard
pen = turtle.Turtle()
pen.speed(0)
pen.color("white")
pen.penup()
pen.hideturtle()
pen.goto(0, 260)
pen.write("Score: 0", align="center", font=("Courier", 18, "normal"))

# Movement


def go_left():
    x = bucket.xcor()
    if x > -250:
        x -= 30
        bucket.setx(x)


def go_right():
    x = bucket.xcor()
    if x < 250:
        x += 30
        bucket.setx(x)


# Keyboard bindings
wn.listen()
wn.onkeypress(go_left, "Left")
wn.onkeypress(go_right, "Right")

# Game loop
while True:
    wn.update()

    # Move the ball down
    y = ball.ycor()
    y -= 20
    ball.sety(y)

    # Check if the ball hits the bottom
    if ball.ycor() < -290:
        ball.goto(random.randint(-250, 250), 250)
        ball.color(random.choice(colors))
        # Decrease score for missing
        score -= 2

    # Check for collision
    if abs(ball.ycor() - bucket.ycor()) < 30 and abs(ball.xcor() - bucket.xcor()) < 50:
        if ball.color()[0] == "red":
            score += 5
        elif ball.color()[0] == "blue":
            score += 10
        elif ball.color()[0] == "green":
            score += 15
        elif ball.color()[0] == "yellow":
            score += 20
        else:
            score += 25

        ball.goto(random.randint(-250, 250), 250)
        ball.color(random.choice(colors))
        speed = max(0.01, speed - 0.001)  # Increase difficulty

    # Update the score display
    pen.clear()
    pen.write(f"Score: {score}", align="center",
              font=("Courier", 18, "normal"))

    time.sleep(speed)
