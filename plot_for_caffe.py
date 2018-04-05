"""
Plot graph from caffe output log.
It can send figure and optimization done message to your slack messanger.

To plot graph, caffe output log is parsed by regular expression.
The regular expression is defiined at plot() function.

If you want to use slack alert,
you must make 'slack_setup.bin' file.
You can make the file using 
SlackHandler.set_info_to_setup_file('your bot token', 'your channel')
"""
from __future__ import print_function 

import sys, os
import re, argparse, subprocess
from datetime import datetime, timedelta
import logging, pickle

from matplotlib import pyplot as plt
from matplotlib import animation
from matplotlib import style


LOSS_MIN = 0.1
LOSS_MAX = 0.3
STEP_MAX = 90000
LR_MULT = 10000 # learning rate multiplyer

logger = logging.getLogger('')
fomatter = logging.Formatter('[%(levelname)s|%(filename)s:%(lineno)s] %(asctime)s > %(message)s')
streamHandler = logging.StreamHandler()
streamHandler.setFormatter(fomatter)
logger.addHandler(streamHandler)


class Subplot(object):
    def __init__(self, ax=None, fig=None, 
                 x_mult=1, y_mult=1, x_regex=None, y_regex=None, 
                 xlabel=None, ylabel=None, 
                 xmin=0, xmax=None, ymin=0, ymax=None):
        if ax is not None:
            if fig is not None:
                raise ValueError('ax and fig cannot be used together')
            self.ax = ax.twinx()
        else:
            if fig is None:
                raise ValueError('ax or fig is needed')
            self.ax = fig.add_subplot(1, 1, 1)
        if x_regex is None:
            raise ValueError('x_regex is needed')
        if y_regex is None:
            raise ValueError('y_regex is needed')
        self.x_mult = x_mult
        self.y_mult = y_mult
        self.ylabel = ylabel
        if xlabel is not None:
            self.ax.set_xlabel(xlabel)
        if ylabel is not None:
            if y_mult != 1:
                ylabel = ylabel + " (x%s)" % y_mult
            self.ax.set_ylabel(ylabel)
        if xmax is not None:
            self.ax.set_xlim(xmin, xmax)
        if ymin is not None and ymax is not None:
            self.ax.set_ylim(ymin, ymax)

        self.x_regex = x_regex
        self.y_regex = y_regex

        self.auxiliary_line_dict = dict()
    
    def init_target_line(self, log_file_path, line_type=None, label_header=''):
        if type(label_header) is not str:
            raise ValueError('label_header should be str type') 
        label = label_header + self.ylabel
        plot_, = self.ax.plot([], [], line_type, label=label)
        self.target_line = Line(plot_, log_file_path, self.x_regex, self.y_regex, self.x_mult, self.y_mult, line_type, label)
        self.target_label = label

    def add_auxiliary_line(self, log_file_path, line_type='--', label_header=''):
        if type(label_header) is not str:
            raise ValueError('label_header should be str type') 
        label = label_header + self.ylabel
        plot_, = self.ax.plot([], [], line_type, label=label)
        self.auxiliary_line_dict[label] = Line(plot_, log_file_path, self.x_regex, self.y_regex, self.x_mult, self.y_mult, line_type, label)
    
    def get_all_lines(self):
        lines = [self.target_line] + self.auxiliary_line_dict.values()
        return [x.plot for x in lines]
    
    def get_all_labels(self):
        return [self.target_label] + self.auxiliary_line_dict.keys()


class Line(object):
    def __init__(self, plot, log_file_path, x_regex, y_regex, x_mult=1, y_mult=1, line_type=None, label=None):
        self.log_file_parser = LogFileParser(log_file_path)
        self.log_file_parser.update_logs()
        self.x_regex = x_regex
        self.y_regex = y_regex
        self.x_mult = x_mult
        self.y_mult = y_mult
        self.line_type = line_type
        self.label = label
        self.plot = plot

    def draw(self):
        self.log_file_parser.update_logs()
        x_list = self.log_file_parser.parse_to_list(self.x_regex)
        y_list = self.log_file_parser.parse_to_list(self.y_regex)
        x_list = [x*self.x_mult for x in x_list]
        y_list = [y*self.y_mult for y in y_list]
        if len(x_list) in [len(y_list), len(y_list)+1, len(y_list)-1]:
            min_len = len(x_list) if len(x_list) < len(y_list) else len(y_list)
            x_list = x_list[:min_len]
            y_list = y_list[:min_len]
        else:
            raise ValueError("length of x(%s) != length of y(%s)" % (len(x_list), len(y_list)))
        self.plot.set_data(x_list, y_list)
    
    def latest_x(self):
        return self.log_file_parser.parse_to_list(self.x_regex)[-1]


class LogFileParser(object):
    def __init__(self, log_file_path):
        self.log_file_path = log_file_path
        self.update_logs()
    
    def update_logs(self):
        with open(self.log_file_path, 'r') as f:
            self.logs = f.read()
    
    def find(self, regex):
        self.update_logs()
        regex = re.compile(regex)
        return regex.search(self.logs)
        
    def parse_to_list(self, regex):
        '''
        Find regex in the log file,
        and extract only float number from regex result.
        finally return list of float number.
        '''
        self.update_logs()
        regex = re.compile(regex)
        regex_iter = regex.finditer(self.logs)
        return [self.__line_to_float(x.group()) for x in regex_iter]

    def __line_to_float(self, text):
        re_float_number = re.compile(r"[\d.]+(e[-+][\d]{2}){0,1}")
        return float(re_float_number.search(text).group())
    
    def time_of_n_th_step(self, step):
        self.update_logs()
        re_step = re.compile(r".* Iteration %s, loss" % step)
        re_time = re.compile(r"\d{4} \d{2}:\d{2}:\d{2}")
        raw_time = re_time.search(re_step.search(self.logs).group()).group()
        step_time = datetime.strptime(raw_time, "%m%d %H:%M:%S")
        step_time = step_time.replace(year=datetime.now().year)
        if step_time > datetime.now():
            step_time = datetime.replace(year=step_time.year-1)
        return step_time
        

class SlackHandler(object):
    def __init__(self, slack_alert):
        self.is_active = bool(slack_alert)
        self.alert_interval = slack_alert
        self.setup_file = "slack_setup.bin"
        self.get_info_from_setup_file()
        self.last_fid = None
    
    def get_info_from_setup_file(self):
        with open(self.setup_file, 'rb') as f:
            self.bot_token, self.channel = pickle.load(f)
    
    def set_info_to_setup_file(self, bot_token, channel):
        '''
        make slack_setup.bin file.
        Input
            bot_token: your bot token issued from slack.
            channel: name of the channel where your message would be uploaded.
        '''
        with open(self.setup_file, 'wb') as f:
            pickle.dump((bot_token, channel), f)
    
    def deactivate(self):
        self.is_active = False

    def send_message(self, title, body):
        def _send_message(title, body):
            try:
                result = subprocess.check_output("curl -F token=%s -F channel=#%s -F text=\"*%s* %s\" https://slack.com/api/chat.postMessage"
                                            % (self.bot_token, self.channel, title, body), stderr=subprocess.STDOUT, shell=True)
            except subprocess.CalledProcessError:
                logger.warning("cannot send message to slack")
                result = ''
            else:
                if not re.search(r'"ok":true', result):
                    logger.warning("cannot send message to slack")
            return result
        result = _send_message(title, body)

    def send_figure(self, fig, graph_title, body=''):
        temp_figure_image_name = "tmp_fig.png"
        fig.savefig(temp_figure_image_name)
        try:
            self.send_image(graph_title, temp_figure_image_name, body)
        except AssertionError as e:
            print(e)
        os.remove(temp_figure_image_name)

    def send_image(self, title, image_path, body=''):
        def _send_image(title, image_path, body=''):
            try:
                result = subprocess.check_output("curl -F token=%s -F channels=#%s -F title='%s' -F initial_comment='%s' -F file=@%s https://slack.com/api/files.upload"
                                            % (self.bot_token, self.channel, title, body, image_path), stderr=subprocess.STDOUT, shell=True)
            except subprocess.CalledProcessError:
                logger.warning("cannot send image to slack")
                result = ''
            else:
                if not re.search(r'"ok":true', result):
                    logger.warning("cannot send image to slack")
            return result
        
        def _delete_last_image(fid):
            try:
                result = subprocess.check_output("curl -F token=%s -F file=%s https://slack.com/api/files.delete" % (self.bot_token, fid), shell=True)
            except subprocess.CalledProcessError:
                logger.warning("cannot delete last image")
                result = ''
            else:
                if not re.search(r'"ok":true', result):
                    logger.warning("cannot delete last image: '%s'" % result)
            return result

        def _get_file_id(result):
            targetline = re.search(r'"id":"\w+"', result).group()
            return re.findall(r'\w+', targetline)[-1]
        
        result = _send_image(title, image_path, body)
        if self.last_fid is not None:
            _delete_last_image(self.last_fid)
        try:
            self.last_fid = _get_file_id(result)
        except:
            logger.warning("cannot send image to slack")


class TimeCalculator(object):
    def __init__(self, interval, max_step, log_file_path):
        self.stepdelta = 1000
        self.datetime_format = "%Y-%m-%d %H:%M:%S"
        self.max_step = max_step
        self.interval = timedelta(minutes=interval)
        self.log_file_path = log_file_path
    
    def estimate_remaining_time(self, current_step):
        current_step = int(current_step)
        last_step = current_step - self.stepdelta if current_step > self.stepdelta else 0
        time_consumption = self.calculate_time_comsumption(last_step, current_step)
        remaining_time = int((self.max_step - current_step) / (current_step - last_step)) * time_consumption
        return remaining_time

    def estimate_end_time(self, current_step):
        return (datetime.now() + self.estimate_remaining_time(current_step)).strftime(self.datetime_format)

    def calculate_time_comsumption(self, start_step, end_step):
        start_step = int(start_step)
        end_step = int(end_step)
        end_time = LogFileParser(self.log_file_path).time_of_n_th_step(end_step)
        start_time = LogFileParser(self.log_file_path).time_of_n_th_step(start_step)
        return end_time - start_time

    def get_start_time(self):
        return LogFileParser(self.log_file_path).time_of_n_th_step(0).strftime(self.datetime_format)
        

class MinutesTimer(object):
    def __init__(self, interval):
        self.interval = timedelta(minutes=interval)
        self.time_to_go = datetime.now()
    
    def is_active(self):
        if self.time_to_go < datetime.now():
            self.time_to_go += self.interval
            return True
        return False


class LineColorCycler(object):
    """
    Choose matplotlib line color automatically.
    There are 10 colors; C0 to C9.
    After using C9, It use C0.
    """
    def __init__(self):
        self.__num_of_colors = 10
        self.__color_list = ['C%s' % i for i in range(self.__num_of_colors)]
        self.__current_index = 1

    def __call__(self):
        return self.__color_list[self.__get_current_index()]

    def __get_current_index(self):
        current_index = self.__current_index
        self.__current_index = (self.__current_index + 1) % self.__num_of_colors
        return current_index


line_color_cycler = LineColorCycler()


def main():
    PARSER = argparse.ArgumentParser()
    PARSER.add_argument("log_file_path", type=str, help="path to the log file to plot")
    PARSER.add_argument("-c", "--comparision", type=str, default=None, help="path to the comparision log file to plot, default: not plot")
    PARSER.add_argument("-g", "--graph_title", type=str, default=None, help="title of the graph, default: the log file's name")
    PARSER.add_argument("-i", "--interval", type=int, default=10000, help="plotting interval(ms), default: 10000")
    PARSER.add_argument("-d", "--dark_mode", action="store_true", help="turns on the dark mode")
    PARSER.add_argument("-q", "--auto_quit", action="store_true", help="quit automatically after optimization done")
    PARSER.add_argument("-s", "--slack_alert", type=int, default=0, help="send message after optimization done or figure every SLACK_ALERT minutes to slack, default is 0(not send)")
    ARGS = PARSER.parse_args()
    
    if not os.path.isfile(ARGS.log_file_path):
        logger.error("there is no log file at '%s'" % ARGS.log_file_path)
        raise TypeError
    graph_title = ARGS.graph_title
    if graph_title is None:
        graph_title = os.path.basename(ARGS.log_file_path).replace(".log", "")
    if ARGS.dark_mode:
        style.use('dark_background')
    
    plot(ARGS.log_file_path, ARGS.comparision, graph_title, ARGS.slack_alert, ARGS.auto_quit, ARGS.interval)


def plot(log_file_path, comparision_log_file_path, graph_title, slack_alert, auto_quit, refresh_interval, lr_mult=LR_MULT):
    fig = plt.figure()
    subplot_dict = dict()
    subplot_dict['loss'] = Subplot(fig=fig, xlabel="iteration", xmax=STEP_MAX, ylabel="loss", ymin=LOSS_MIN, ymax=LOSS_MAX, 
                            x_regex=r"Iteration [\d]+, loss", y_regex=r", loss = [\d.]+(e[-+][\d]{2}){0,1}")
    subplot_dict['lr'] = Subplot(ax=subplot_dict['loss'].ax, y_mult=lr_mult, ylabel='lr', ymin=0, ymax=1, 
                            x_regex=r"Iteration [\d]+, lr", y_regex=r"lr = [\d.]+(e[-+][\d]{2}){0,1}")
    
    for _subplot in subplot_dict.values():
        _subplot.init_target_line(log_file_path=log_file_path, line_type=line_color_cycler())

    plot_comparision(subplot_dict, comparision_log_file_path, label_header="comp_")
    setup_legend(subplot_dict)
    
    slack_handler = SlackHandler(slack_alert)
    n_minutes_timer = MinutesTimer(slack_handler.alert_interval)
    time_calc = TimeCalculator(interval=slack_handler.alert_interval, max_step=STEP_MAX, log_file_path=log_file_path)

    def plot_iteratively():
        def animate(frame):
            for _subplot in subplot_dict.values():
                _subplot.target_line.draw()
            if slack_handler.is_active and n_minutes_timer.is_active():
                message = write_message_body(subplot_dict.values()[0], time_calc)
                fig = plt.gcf()
                slack_handler.send_figure(fig, graph_title, body=message)
            if is_optimization_done(log_file_path):
                if slack_handler.is_active:
                    fig = plt.gcf()
                    slack_handler.send_figure(fig, graph_title)
                    slack_handler.send_message(graph_title, 'optimization done')
                    slack_handler.deactivate()
                if auto_quit:
                    sys.exit()

        return animation.FuncAnimation(fig, animate, interval=refresh_interval)
    
    ani = plot_iteratively()
    
    plt.title(graph_title, fontsize=15)
    plt.grid(True)
    plt.subplots_adjust(left=0.12, right=0.85)
    plt.show()


def plot_comparision(subplot_dict, comparision_log_file_path, label_header):
    try:
        assert os.path.isfile(comparision_log_file_path), "no comparision file at '%s'" % comparision_log_file_path        
    except AssertionError as e:
        logger.warning(e)
    except TypeError: # no comparision option
        pass
    else:
        for _subplot in subplot_dict.values():
            _subplot.add_auxiliary_line(log_file_path=comparision_log_file_path, line_type=line_color_cycler() + '--', label_header=label_header)
            for line in _subplot.auxiliary_line_dict.values():
                line.draw()


def write_message_body(subplot, time_calc):
    latest_step = subplot.target_line.latest_x()
    start_time = time_calc.get_start_time()
    end_time = time_calc.estimate_end_time(latest_step)
    time_consumption = time_calc.calculate_time_comsumption(0, latest_step)
    remaining_time = time_calc.estimate_remaining_time(latest_step)
    message = "Started at\t%s\nWill end at\t%s\nConsumped\t%s\nRemain\t%s" \
                % (start_time, end_time, time_consumption, remaining_time)
    return message


def setup_legend(subplot_dict):
    lines, labels = [], []
    for _subplot in subplot_dict.values():
        lines += _subplot.get_all_lines()
        labels += _subplot.get_all_labels() 
    if len(lines) != len(labels):
        logger.error("the number of lines to plot and labels are different, # of lines: '%s', # of labels: '%s'" % (len(lines), len(labels)))
        raise AssertionError
    plt.legend(lines, labels)


def is_optimization_done(log_file_path):
    log_file_parser = LogFileParser(log_file_path)
    if log_file_parser.find(r"Optimization Done."):
        return True
    return False


if __name__=="__main__":
    main()
