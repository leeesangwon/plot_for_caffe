from __future__ import print_function 

import sys, os
import re, argparse

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
    parser.add_argument("-c", "--comparision", type=str, default=None, help="path to the comparision log file to plot, default: not plot")
    parser.add_argument("-g", "--graph_title", type=str, default=None, help="title of the graph, default: the log file's name")
    parser.add_argument("-i", "--interval", type=int, default=3000, help="plotting interval(ms), default: 3000")
    parser.add_argument("-d", "--dark_mode", action="store_true", help="turns on the dark mode")
    args = parser.parse_args()

    log_file_path = args.log_file_path
    assert os.path.isfile(log_file_path), "there is no log file at %s" % log_file_path
    graph_title = args.graph_title
    draw_rate = args.interval
    comparision_log_file_path = args.comparision
    if args.dark_mode:
        style.use('dark_background')
    if graph_title is None:
        graph_title = os.path.basename(log_file_path)
    
    plot(log_file_path, comparision_log_file_path, graph_title, draw_rate)


def plot(log_file_path, comparision_log_file_path, graph_title, draw_rate, lr_mult=LR_MULT):
    fig = plt.figure()
    ax1, ax2 = draw_init(fig, lr_mult)
    
    loss_plot, = ax1.plot([], [], 'C1', label="total_loss")
    lr_plot, = ax2.plot([], [], 'C2', label='learning_rate')
    
    lines = [loss_plot, lr_plot]
    labels = ["loss", "lr"]
    try:
        assert os.path.isfile(comparision_log_file_path), "no comparision file at %s" % comparision_log_file_path
        comparision_loss_plot, = ax1.plot([], [], 'C3--', label="comparision_loss")
        comparision_lr_plot, = ax2.plot([], [], 'C4--', label="comparision_lr")

        lines.append(comparision_loss_plot)
        labels.append("comp_loss")
        lines.append(comparision_lr_plot)
        labels.append("comp_lr")

        def plot_comparision():
            with open(comparision_log_file_path, 'r') as log_file:
                lines = log_file.read()
            draw_once(lines, comparision_loss_plot, comparision_lr_plot, lr_mult)
        
        plot_comparision()
    except AssertionError as e:
        print(e)
    except TypeError:
        pass

    assert len(lines) == len(labels), "the number of lines to plot and labels are different, # of lines: %s, # of labels: %s" % (len(lines), len(labels))
    plt.legend(lines, labels)

    def plot_iteratively():
        def animate(frame):
            with open(log_file_path, 'r') as log_file:
                lines = log_file.read()
            draw_once(lines, loss_plot, lr_plot, lr_mult)
            if is_optimization_done(lines):
                sys.exit()

        return animation.FuncAnimation(fig, animate, interval=draw_rate)
    
    ani = plot_iteratively()

    plt.title(graph_title, fontsize=15)
    plt.grid(True)
    plt.subplots_adjust(left=0.12, right=0.85)
    plt.show()


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

def draw_once(lines, loss_plot, lr_plot, lr_mult):
    iter_list, loss_list, lr_list, _, _ = parse_to_list(lines)
    loss_plot.set_data(iter_list, loss_list)
    lr_list = [x*lr_mult for x in lr_list]
    lr_plot.set_data(iter_list, lr_list)
    

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
