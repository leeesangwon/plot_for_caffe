from __future__ import print_function 

import sys, os
import re, argparse, subprocess
from datetime import datetime, timedelta

from matplotlib import pyplot as plt
from matplotlib import animation
from matplotlib import style

# for slackbot
CHANNEL = 'change to your channel'
BOT_TOKEN = "change to your bot token"
temp_figure_image_name = "tmp_fig.png"

LOSS_MIN = 0.1
LOSS_MAX = 0.3
STEP_MAX = 90000
LR_MULT = 10000 # learning rate multiplyer

PARSER = argparse.ArgumentParser()
PARSER.add_argument("log_file_path", type=str, help="path to the log file to plot")
PARSER.add_argument("-c", "--comparision", type=str, default=None, help="path to the comparision log file to plot, default: not plot")
PARSER.add_argument("-g", "--graph_title", type=str, default=None, help="title of the graph, default: the log file's name")
PARSER.add_argument("-i", "--interval", type=int, default=10000, help="plotting interval(ms), default: 10000")
PARSER.add_argument("-d", "--dark_mode", action="store_true", help="turns on the dark mode")
PARSER.add_argument("-q", "--auto_quit", action="store_true", help="quit automatically after optimization done")
PARSER.add_argument("-s", "--slack_alert", type=int, default=0, help="send message after optimization done or figure every SLACK_ALERT minutes to slack, default is 0(not send)")
ARGS = PARSER.parse_args()

def main():
    log_file_path = ARGS.log_file_path
    assert os.path.isfile(log_file_path), "there is no log file at %s" % log_file_path
    graph_title = ARGS.graph_title
    comparision_log_file_path = ARGS.comparision
    slack_alert = ARGS.slack_alert
    if ARGS.dark_mode:
        style.use('dark_background')
    if graph_title is None:
        graph_title = os.path.basename(log_file_path).replace(".log", "")
    
    plot(log_file_path, comparision_log_file_path, graph_title, slack_alert)


def plot(log_file_path, comparision_log_file_path, graph_title, slack_alert, lr_mult=LR_MULT):
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
            if ARGS.slack_alert and n_minutes_timer(ARGS.slack_alert):
                fig = plt.gcf()
                fig.savefig(temp_figure_image_name)
                send_figure_to_slack(graph_title, temp_figure_image_name)
                os.remove(temp_figure_image_name)
            if is_optimization_done(lines):
                if ARGS.slack_alert:
                    # send figure to slack
                    fig = plt.gcf()
                    fig.savefig(temp_figure_image_name)
                    send_figure_to_slack(graph_title, temp_figure_image_name)
                    os.remove(temp_figure_image_name)
                    # send message to slack
                    send_message_to_slack(graph_title, 'optimization done')
                    # not to send message and figure iterativly
                    ARGS.slack_alert = 0
                if ARGS.auto_quit:
                    sys.exit()

        return animation.FuncAnimation(fig, animate, interval=ARGS.interval)
    
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


def send_message_to_slack(title, body):
    def send_message(title, body):
        result = subprocess.check_output("curl -F token=%s -F channel=#%s -F text=\"*%s* %s\" https://slack.com/api/chat.postMessage"
                                         % (BOT_TOKEN, CHANNEL, title, body), shell=True)
        assert re.search(r'"ok":true', result), "cannot send message to slack"
        return result
    
    result = send_message(title, body)


def send_figure_to_slack(title, figure_name):
    def send_figure(title, figure_name):
        result = subprocess.check_output("curl -F token=%s -F channels=#%s -F title=%s -F file=@%s https://slack.com/api/files.upload"
                                         % (BOT_TOKEN, CHANNEL, title, figure_name), shell=True)
        assert re.search(r'"ok":true', result), "cannot send figure to slack"
        return result
    
    def delete_last_figure(fid):
        result = subprocess.check_output("curl -F token=%s -F file=%s https://slack.com/api/files.delete" % (BOT_TOKEN, fid), shell=True)
        assert re.search(r'"ok":true', result), "cannot delete last figure: %s" % result

    def get_file_id(result):
        targetline = re.search(r'"id":"\w+"', result).group()
        return re.findall(r'\w+', targetline)[-1]
    
    result = send_figure(title, figure_name)
    try:
        delete_last_figure(send_figure_to_slack.last_fid)
    except AttributeError:
        pass
    finally:
        send_figure_to_slack.last_fid = get_file_id(result)


def n_minutes_timer(n):
    try:
        is_time_to_go = n_minutes_timer.time_to_go < datetime.now()
    except AttributeError:
        n_minutes_timer.time_to_go = datetime.now()
        is_time_to_go = True
    if is_time_to_go:
        n_minutes_timer.time_to_go += timedelta(minutes=n)
    return is_time_to_go


if __name__=="__main__":
    main()
