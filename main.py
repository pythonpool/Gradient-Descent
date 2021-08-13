#!/usr/bin/python
# -*- coding: utf-8 -*-
from numpy import *


def compute_error(b, m, points):
    totalError = 0
    for i in range(0, len(points)):
        x = points[i, 0]
        y = points[i, 1]
        totalError += (y - (m * x + b)) ** 2
    return totalError / float(len(points))


def gradient_step(
    b_current,
    m_current,
    points,
    learningRate,
    ):
    b_gradient = 0
    m_gradient = 0
    N = float(len(points))
    for i in range(0, len(points)):
        x = points[i, 0]
        y = points[i, 1]
        b_gradient += -(2 / N) * (y - (m_current * x + b_current))
        m_gradient += -(2 / N) * x * (y - (m_current * x + b_current))
    new_b = b_current - learningRate * b_gradient
    new_m = m_current - learningRate * m_gradient
    return [new_b, new_m]


def gradient_descent_runner(
    points,
    starting_b,
    starting_m,
    learning_rate,
    iterations,
    ):
    b = starting_b
    m = starting_m
    for i in range(iterations):
        (b, m) = gradient_step(b, m, array(points), learning_rate)
    return [b, m]


def main():
    points = genfromtxt('data.csv', delimiter=',')
    learning_rate = 0.00001
    initial_b = 0
    initial_m = 0
    iterations = 10000
    print('Starting gradient descent at b = {0}, m = {1}, error = {2}'.format(initial_b,
            initial_m, compute_error(initial_b, initial_m, points)))
    print('Running...')
    [b, m] = gradient_descent_runner(points, initial_b, initial_m,
            learning_rate, iterations)
    print('After {0} iterations b = {1}, m = {2}, error = {3}'.format(iterations,
            b, m, compute_error(b, m, points)))


if __name__ == '__main__':
    main()
