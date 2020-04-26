import numpy as np
import matplotlib.pyplot as plt
np.random.seed(2)
# DATA #

x = np.array([-2, -1, 25, 6, 17, 4, -15, -6]).reshape((4, 2))

y = np.array([1, 0, 0, 1]).reshape((4, 1))

# Sigmoid function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Derivative function

def sig_derivative(x):  # x=sigmoid(x)
    return x * (1 - x)


def change_in_theta_of_layer1():
    plt.scatter(i, np.mean(np.abs(w1[:, [0]])), color='red')
    plt.scatter(i, np.mean(np.abs(w1[:, [1]])), color='blue')
    plt.scatter(i, np.mean(np.abs(w1[:, [2]])), color='y')
    plt.scatter(i, np.mean(np.abs(w1[:, [3]])), color='green')
    plt.draw()
    plt.pause(0.00000001)


def change_in_error_with_itterationes():
    plt.scatter(i, np.mean(np.abs(error_3)), c='b')
    plt.scatter(i, np.mean(np.abs(error_2)), c='r')
    plt.xlabel('NO OF ITERATIONS')
    plt.ylabel('ERROR(LAYER1, LAYER2')
    plt.draw()
    plt.pause(0.00000001)


def change_in_pridction():
    plt.scatter(np.array([1, 2, 3, 4]), y, c='r')
    plt.scatter(np.array([1, 2, 3, 4]), l3, c='b')

    plt.draw()

    plt.pause(0.00000001)


w1 = 2 * np.random.random((2, 4)) - 1

w2 = 2 * np.random.random(((4, 1))) - 1

AA=[]
for i in range(2000):
    # FORWARD PROPAGATION
    l1 = x

    l2 = sigmoid(np.dot(l1, w1))

    l3 = sigmoid(np.dot(l2, w2))

    error_3 = y - l3

    # BACKWARD PROGATION
    if (i % 100) == 0:
        print(i)
        print("Error:" + str(np.mean(np.abs(error_3))))

    delta_3 = error_3 * sig_derivative(l3)
    error_2 = delta_3.dot(w2.T)

    delta_2 = error_2 * sig_derivative(l2)

    w2 = w2 + l2.T.dot(delta_3)
    w1 = w1 + l1.T.dot(delta_2)

    aab = np.sum(np.array(y - l3) ** 2)
    AA.append(aab)
    if i % 10 == 0:
        ## UNHASH THIS FUNCTION TO VIZVALIZE THE CHANGES  ##

        #change_in_error_with_itterationes()
        change_in_theta_of_layer1()

        #change_in_pridction()

plt.show()

''' TO SHOW THE COST FUNCTION '''

plt.plot(np.linspace(1,len(AA),len(AA)),AA)
plt.xlabel('ITTERATIONES')
plt.ylabel('COST FUNCTION VALUES')
plt.show()

print(l3)


