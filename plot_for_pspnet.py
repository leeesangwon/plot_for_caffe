import os
import re
import sys
import argparse

from matplotlib import pyplot as plt
from matplotlib import animation
from matplotlib import style

LOSS_MIN = 0.1
LOSS_MAX = 0.3
STEP_MAX = 90000
LR_MULT = 10000 # learning rate multiplyer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("log_file_path", type=str, help="path to the log file to plot")
    parser.add_argument("-g", "--graph_title", type=str, default=None, help="title of the graph, default: the log file's name")
    parser.add_argument("-i", "--interval", type=int, default=3000, help="plotting interval, unit: ms, default: 3000")
    parser.add_argument("-d", "--dark_mode", action="store_true", help="dark mode turns on")
    args = parser.parse_args()

    log_file_path = args.log_file_path
    graph_title = args.graph_title
    draw_rate = args.interval
    if args.dark_mode:
        style.use('dark_background')
    if graph_title is None:
        graph_title = os.path.basename(log_file_path)
    plot(log_file_path, graph_title, draw_rate)


def plot(log_file_path, graph_title, draw_rate, lr_mult=LR_MULT):
    fig = plt.figure()
    ani = plot_iteratively(fig, log_file_path, draw_rate, lr_mult)

    plt.title(graph_title, fontsize=15)
    plt.grid(True)
    plt.subplots_adjust(left=0.12, right=0.85)
    plt.show()


def plot_iteratively(fig, log_file_path, draw_rate, lr_mult):
    ax1, ax2 = draw_init(fig, lr_mult)

    def animate(i, *fargs):
        with open(log_file_path, 'r') as log_file:
            lines = log_file.read()
        draw_once(lines, ax1, ax2, lr_mult)
        if is_optimization_done(lines):
            sys.exit()

    return animation.FuncAnimation(fig, animate, interval=draw_rate)


def draw_init(fig, lr_mult, xmax=STEP_MAX, loss_min=LOSS_MIN, loss_max=LOSS_MAX):
    ax1 = fig.add_subplot(1, 1, 1)
    ax1.set_xlim(0, xmax)
    ax1.set_ylabel("Loss")
    ax1.set_ylim(loss_min, loss_max)
    ax1.set_xlabel("iteration")
    ax2 = ax1.twinx()
    ax2.set_ylabel("Learning rate (x%s)" % lr_mult)
    ax2.set_ylim(0, 1)

    return ax1, ax2

def draw_once(lines, ax1, ax2, lr_mult, color_loss='r', color_lr='g'):
    iter_list, loss_list, lr_list, aux1_list, aux2_list = parse_to_list(lines)
     
    loss_plot, = ax1.plot(iter_list, loss_list, color_loss, label="total_loss")
    
    lr_list = [x*lr_mult for x in lr_list]
    lr_plot, = ax2.plot(iter_list, lr_list, color_lr, label="learning_rate")
    
    plt.legend([loss_plot, lr_plot], ["loss", "lr"])


def parse_to_list(lines):
    re_iteration_line = re.compile(r"Iteration [\d]+, loss")
    re_total_loss_line = re.compile(r", loss = [\d.]+(e[-+][\d]{2}){0,1}")
    re_learning_rate_line = re.compile(r"lr = [\d.]+(e[-+][\d]{2}){0,1}")
    re_aux_loss_line_1 = re.compile(r"loss_auxiliary = [\d.]+(e[-+][\d]{2}){0,1}")
    re_aux_loss_line_2 = re.compile(r"loss_main = [\d.]+(e[-+][\d]{2}){0,1}")
    
    iteration_line = re_iteration_line.finditer(lines)
    total_loss_line = re_total_loss_line.finditer(lines)
    learning_rate_line = re_learning_rate_line.finditer(lines)
    aux_loss_line_1 = re_aux_loss_line_1.finditer(lines)
    aux_loss_line_2 = re_aux_loss_line_2.finditer(lines)

    iter_list = []
    loss_list = []
    lr_list  = []
    aux1_list = []
    aux2_list = []

    for iter_num, loss, lr, aux1, aux2 in zip(
        iteration_line, total_loss_line, learning_rate_line, aux_loss_line_1, aux_loss_line_2):
        try:
            iter_list.append(line_to_float(iter_num.group()))
            loss_list.append(line_to_float(loss.group()))
            lr_list.append(line_to_float(lr.group()))
            aux1_list.append(line_to_float(aux1.group()))
            aux2_list.append(line_to_float(aux2.group()))
        except ValueError:
            pass
    
    return iter_list, loss_list, lr_list, aux1_list, aux2_list
    

def line_to_float(text):
    re_float_number = re.compile(r"[\d.]+(e[-+][\d]{2}){0,1}")
    return float(re_float_number.search(text).group())


def is_optimization_done(lines):
    re_optimization_done = re.compile(r"Optimization Done.")
    if re_optimization_done.search(lines):
        return True
    return False


if __name__=="__main__":
    main()
