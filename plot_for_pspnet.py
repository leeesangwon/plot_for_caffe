import os
import re
import sys
import matplotlib.pyplot as plt

LOSS_MIN = 0.1
LOSS_MAX = 0.3
STEP_MAX = 90000
DRAW_RATE = 3 # secs


def main(log_file_path, graph_title=None):
    if graph_title is None:
        graph_title = os.path.basename(log_file_path)
    draw(log_file_path, graph_title)


def draw(log_file_path, graph_title, xmax=STEP_MAX, loss_min=LOSS_MIN, loss_max=LOSS_MAX, draw_rate=DRAW_RATE, lr_mult=10000):
    plt.ion()
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.set_ylabel("Loss")
    ax1.set_ylim(loss_min, loss_max)
    ax1.set_xlim(0, xmax)
    bx1 = ax1.twinx()
    bx1.set_ylabel("Learning rate (x%s)" % lr_mult)
    
    while(True):
        with open(log_file_path, 'r') as log_file:
            lines = log_file.read()
        
        draw_once(lines, graph_title, ax1, bx1, lr_mult)
        plt.pause(draw_rate)
        
        if is_optimization_done(lines):
            sys.exit()


def draw_once(lines, graph_title, ax1, bx1, lr_mult):
    iter_list, loss_list, lr_list, aux1_list, aux2_list = parse_to_list(lines)
     
    loss_plot, = ax1.plot(iter_list, loss_list, 'r', label="total_loss")

    lr_list = [x*lr_mult for x in lr_list]
    lr_plot, = bx1.plot(iter_list, lr_list, 'g', label="learning_rate")
    
    plt.legend([loss_plot, lr_plot], ["loss", "lr"])
    plt.title(graph_title)
    plt.xlabel("iteration")
    plt.grid(True)
    plt.subplots_adjust(left=0.12, right=0.85)

    plt.show()


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
    if len(sys.argv) == 1:
        DIR = os.path.join(os.getcwd(), "pspnet473_cityscapes_iter90000.log")
    else:
        DIR = sys.argv[1]
    main(DIR)
