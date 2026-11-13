import time
import copy
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import FancyArrowPatch
from matplotlib.widgets import Button
from matplotlib import cm



dlc = dict(dlblue = '#0096ff', dlorange = '#FF9300', dldarkred='#C00000', dlmagenta='#FF40FF', dlpurple='#7030A0')
dlblue = '#0096ff'; dlorange = '#FF9300'; dldarkred='#C00000'; dlmagenta='#FF40FF'; dlpurple='#7030A0'
dlcolors = [dlblue, dlorange, dldarkred, dlmagenta, dlpurple]

def sigmoid(z):
    """
    Compute the sigmoid of z

    Parameters
    ----------
    z : array_like
        A scalar or numpy array of any size.

    Returns
    -------
     g : array_like
         sigmoid(z)
    """
    z = np.clip( z, -500, 500)           # protect against overflow
    g = 1.0/(1.0 + np.exp(-z))

    return g

def log_1pexp(x, maximum=20):
    ''' approximate log(1+exp^x)
        https://stats.stackexchange.com/questions/475589/numerical-computation-of-cross-entropy-in-practice
    Args:
    x   : (ndarray Shape (n,1) or (n,)  input
    out : (ndarray Shape matches x      output ~= np.log(1+exp(x))
    '''

    out  = np.zeros_like(x,dtype=float)
    i    = x <= maximum
    ni   = np.logical_not(i)

    out[i]  = np.log(1 + np.exp(x[i]))
    out[ni] = x[ni]
    return out


def compute_cost_matrix(X, y, w, b, logistic=False, lambda_=0, safe=True):
    """
    Computes the cost using  using matrices
    Args:
      X : (ndarray, Shape (m,n))          matrix of examples
      y : (ndarray  Shape (m,) or (m,1))  target value of each example
      w : (ndarray  Shape (n,) or (n,1))  Values of parameter(s) of the model
      b : (scalar )                       Values of parameter of the model
      verbose : (Boolean) If true, print out intermediate value f_wb
    Returns:
      total_cost: (scalar)                cost
    """
    m = X.shape[0]
    y = y.reshape(-1,1)             # ensure 2D
    w = w.reshape(-1,1)             # ensure 2D
    if logistic:
        if safe:  #safe from overflow
            z = X @ w + b                                                           #(m,n)(n,1)=(m,1)
            cost = -(y * z) + log_1pexp(z)
            cost = np.sum(cost)/m                                                   # (scalar)
        else:
            f    = sigmoid(X @ w + b)                                               # (m,n)(n,1) = (m,1)
            cost = (1/m)*(np.dot(-y.T, np.log(f)) - np.dot((1-y).T, np.log(1-f)))   # (1,m)(m,1) = (1,1)
            cost = cost[0,0]                                                        # scalar
    else:
        f    = X @ w + b                                                        # (m,n)(n,1) = (m,1)
        cost = (1/(2*m)) * np.sum((f - y)**2)                                   # scalar

    reg_cost = (lambda_/(2*m)) * np.sum(w**2)                                   # scalar

    total_cost = cost + reg_cost                                                # scalar

    return total_cost                                                           # scalar


def plt_logistic_squared_error(X, y):
    """ plots logistic squared error for demonstration """
    wx, by = np.meshgrid(np.linspace(-6, 12, 50),
                         np.linspace(10, -20, 40))
    points = np.c_[wx.ravel(), by.ravel()]
    cost = np.zeros(points.shape[0])

    for i in range(points.shape[0]):
        w, b = points[i]
        cost[i] = np.mean((sigmoid(X * w + b) - y)**2) / 2
    cost = cost.reshape(wx.shape)

    fig = plt.figure(figsize=(10, 7.5))
    fig.canvas.toolbar_visible = False
    fig.canvas.header_visible = False
    fig.canvas.footer_visible = False
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    ax.plot_surface(wx, by, cost, alpha=0.6, cmap=cm.coolwarm)

    ax.set_xlabel('w', fontsize=16)
    ax.set_ylabel('b', fontsize=16)
    ax.set_zlabel("Cost", rotation=90, fontsize=16)
    ax.set_title('"Logistic" Squared Error Cost vs (w, b)')
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))

def plt_logistic_cost(X, y):
    """ plots logistic cost """
    wx, by = np.meshgrid(np.linspace(-6,12,50),
                         np.linspace(0, -20, 40))
    points = np.c_[wx.ravel(), by.ravel()]
    cost = np.zeros(points.shape[0], dtype=np.longdouble)

    for i in range(points.shape[0]):
        w,b = points[i]
        cost[i] = compute_cost_matrix(X.reshape(-1, 1), y, w, b, logistic=True, safe=True)
    cost = cost.reshape(wx.shape)

    fig = plt.figure(figsize=(10, 7.5))
    fig.canvas.toolbar_visible = False
    fig.canvas.header_visible = False
    fig.canvas.footer_visible = False
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    ax.plot_surface(wx, by, cost, alpha=0.6, cmap=cm.coolwarm)

    ax.set_xlabel('w', fontsize=16)
    ax.set_ylabel('b', fontsize=16)
    ax.set_zlabel("Cost", rotation=90, fontsize=16)
    ax.set_title('Logistic Cost vs (w, b)')
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))

    return cost    


def compute_gradient_matrix(X, y, w, b, logistic=False, lambda_=0):
    """
    Computes the gradient using matrices

    Args:
      X : (ndarray, Shape (m,n))          matrix of examples
      y : (ndarray  Shape (m,) or (m,1))  target value of each example
      w : (ndarray  Shape (n,) or (n,1))  Values of parameters of the model
      b : (scalar )                       Values of parameter of the model
      logistic: (boolean)                 linear if false, logistic if true
      lambda_:  (float)                   applies regularization if non-zero
    Returns
      dj_dw: (array_like Shape (n,1))     The gradient of the cost w.r.t. the parameters w
      dj_db: (scalar)                     The gradient of the cost w.r.t. the parameter b
    """
    m = X.shape[0]
    y = y.reshape(-1,1)             # ensure 2D
    w = w.reshape(-1,1)             # ensure 2D

    f_wb  = sigmoid( X @ w + b ) if logistic else  X @ w + b      # (m,n)(n,1) = (m,1)
    err   = f_wb - y                                              # (m,1)
    dj_dw = (1/m) * (X.T @ err)                                   # (n,m)(m,1) = (n,1)
    dj_db = (1/m) * np.sum(err)                                   # scalar

    dj_dw += (lambda_/m) * w        # regularize                  # (n,1)

    return dj_db, dj_dw    

def gradient_descent(X, y, w_in, b_in, alpha, num_iters, logistic=False, lambda_=0, verbose=True):
    """
    Performs batch gradient descent to learn theta. Updates theta by taking
    num_iters gradient steps with learning rate alpha

    Args:
      X (ndarray):    Shape (m,n)         matrix of examples
      y (ndarray):    Shape (m,) or (m,1) target value of each example
      w_in (ndarray): Shape (n,) or (n,1) Initial values of parameters of the model
      b_in (scalar):                      Initial value of parameter of the model
      logistic: (boolean)                 linear if false, logistic if true
      lambda_:  (float)                   applies regularization if non-zero
      alpha (float):                      Learning rate
      num_iters (int):                    number of iterations to run gradient descent

    Returns:
      w (ndarray): Shape (n,) or (n,1)    Updated values of parameters; matches incoming shape
      b (scalar):                         Updated value of parameter
    """
    # An array to store cost J and w's at each iteration primarily for graphing later
    J_history = []
    w = copy.deepcopy(w_in)  #avoid modifying global w within function
    b = b_in
    w = w.reshape(-1,1)      #prep for matrix operations
    y = y.reshape(-1,1)

    for i in range(num_iters):

        # Calculate the gradient and update the parameters
        dj_db,dj_dw = compute_gradient_matrix(X, y, w, b, logistic, lambda_)

        # Update Parameters using w, b, alpha and gradient
        w = w - alpha * dj_dw
        b = b - alpha * dj_db

        # Save cost J at each iteration
        if i<100000:      # prevent resource exhaustion
            J_history.append( compute_cost_matrix(X, y, w, b, logistic, lambda_) )

        # Print cost every at intervals 10 times or as many iterations if < 10
        if i% math.ceil(num_iters / 10) == 0:
            if verbose: print(f"Iteration {i:4d}: Cost {J_history[-1]}   ")

    return w.reshape(w_in.shape), b, J_history  #return final w,b and J history for graphing    


class plt_quad_logistic:
  ''' plots a quad plot showing logistic regression '''
  # pylint: disable=too-many-instance-attributes
  # pylint: disable=too-many-locals
  # pylint: disable=missing-function-docstring
  # pylint: disable=attribute-defined-outside-init
  def __init__(self, x_train,y_train, w_range, b_range):
      # setup figure
      fig = plt.figure(figsize=(15, 9))
      fig.canvas.toolbar_visible = False
      fig.canvas.header_visible = False
      fig.canvas.footer_visible = False
      fig.set_facecolor('#ffffff') #white
      gs  = GridSpec(2, 2, figure=fig)
      ax0 = fig.add_subplot(gs[0, 0])
      ax1 = fig.add_subplot(gs[0, 1])
      ax2 = fig.add_subplot(gs[1, 0],  projection='3d')
      ax3 = fig.add_subplot(gs[1,1])
      pos = ax3.get_position().get_points()  ##[[lb_x,lb_y], [rt_x, rt_y]]
      h = 0.05 
      width = 0.2
      axcalc = plt.axes([pos[1,0]-width, pos[1,1]-h, width, h])  #lx,by,w,h
      ax = np.array([ax0, ax1, ax2, ax3, axcalc])
      self.fig = fig
      self.ax = ax
      self.x_train = x_train
      self.y_train = y_train

      self.w = 0. #initial point, non-array
      self.b = 0.

      # initialize subplots
      self.dplot = data_plot(ax[0], x_train, y_train, self.w, self.b)
      self.con_plot = contour_and_surface_plot(ax[1], ax[2], x_train, y_train, w_range, b_range, self.w, self.b)
      self.cplot = cost_plot(ax[3])

      # setup events
      self.cid = fig.canvas.mpl_connect('button_press_event', self.click_contour)
      self.bcalc = Button(axcalc, 'Run Gradient Descent \nfrom current w,b (click)', color=dlc["dlorange"])
      self.bcalc.on_clicked(self.calc_logistic)

#    @output.capture()  # debug
  def click_contour(self, event):
      ''' called when click in contour '''
      if event.inaxes == self.ax[1]:   #contour plot
          self.w = event.xdata
          self.b = event.ydata

          self.cplot.re_init()
          self.dplot.update(self.w, self.b)
          self.con_plot.update_contour_wb_lines(self.w, self.b)
          self.con_plot.path.re_init(self.w, self.b)

          self.fig.canvas.draw()

#    @output.capture()  # debug
  def calc_logistic(self, event):
      ''' called on run gradient event '''
      for it in [1, 8,16,32,64,128,256,512,1024,2048,4096]:
          w, self.b, J_hist = gradient_descent(self.x_train.reshape(-1,1), self.y_train.reshape(-1,1),
                                                np.array(self.w).reshape(-1,1), self.b, 0.1, it,
                                                logistic=True, lambda_=0, verbose=False)
          self.w = w[0,0]
          self.dplot.update(self.w, self.b)
          self.con_plot.update_contour_wb_lines(self.w, self.b)
          self.con_plot.path.add_path_item(self.w,self.b)
          self.cplot.add_cost(J_hist)

          time.sleep(0.3)
          self.fig.canvas.draw()


class data_plot:
  ''' handles data plot '''
  # pylint: disable=missing-function-docstring
  # pylint: disable=attribute-defined-outside-init
  def __init__(self, ax, x_train, y_train, w, b):
      self.ax = ax
      self.x_train = x_train
      self.y_train = y_train
      self.m = x_train.shape[0]
      self.w = w
      self.b = b

      self.plt_tumor_data()
      self.draw_logistic_lines(firsttime=True)
      self.mk_cost_lines(firsttime=True)

      self.ax.autoscale(enable=False) # leave plot scales the same after initial setup

  def plt_tumor_data(self):
      x = self.x_train
      y = self.y_train
      pos = y == 1
      neg = y == 0
      self.ax.scatter(x[pos], y[pos], marker='x', s=80, c = 'red', label="malignant")
      self.ax.scatter(x[neg], y[neg], marker='o', s=100, label="benign", facecolors='none',
                  edgecolors=dlc["dlblue"],lw=3)
      self.ax.set_ylim(-0.175,1.1)
      self.ax.set_ylabel('y')
      self.ax.set_xlabel('Tumor Size')
      self.ax.set_title("Logistic Regression on Categorical Data")

  def update(self, w, b):
      self.w = w
      self.b = b
      self.draw_logistic_lines()
      self.mk_cost_lines()

  def draw_logistic_lines(self, firsttime=False):
      if not firsttime:
          self.aline[0].remove()
          self.bline[0].remove()
          self.alegend.remove()

      xlim  = self.ax.get_xlim()
      x_hat = np.linspace(*xlim, 30)
      y_hat = sigmoid(np.dot(x_hat.reshape(-1,1), self.w) + self.b)
      self.aline = self.ax.plot(x_hat, y_hat, color=dlc["dlblue"],
                                    label="y = sigmoid(z)")
      f_wb = np.dot(x_hat.reshape(-1,1), self.w) + self.b
      self.bline = self.ax.plot(x_hat, f_wb, color=dlc["dlorange"], lw=1,
                                    label=f"z = {np.squeeze(self.w):0.2f}x+({self.b:0.2f})")
      self.alegend = self.ax.legend(loc='upper left')

  def mk_cost_lines(self, firsttime=False):
      ''' makes vertical cost lines'''
      if not firsttime:
          for artist in self.cost_items:
              artist.remove()
      self.cost_items = []
      cstr = f"cost = (1/{self.m})*("
      ctot = 0
      label = 'cost for point'
      addedbreak = False
      for p in zip(self.x_train,self.y_train):
          f_wb_p = sigmoid(self.w*p[0]+self.b)
          c_p = compute_cost_matrix(p[0].reshape(-1,1), p[1],np.array(self.w), self.b, logistic=True, lambda_=0, safe=True)
          c_p_txt = c_p
          a = self.ax.vlines(p[0], p[1], f_wb_p, alpha=0.3, color='k', label=label)
          label='' #just one
          cxy = [p[0], p[1] + (f_wb_p-p[1])/2]
          b = self.ax.annotate(f'{c_p_txt:0.1f}', xy=cxy, xycoords='data',color=dlc["dlpurple"],
                      xytext=(5, 0), textcoords='offset points')
          cstr += f"{c_p_txt:0.1f} +"
          if len(cstr) > 76 and addedbreak is False:
              cstr += "\n"
              addedbreak = True
          ctot += c_p
          self.cost_items.extend((a,b))
      ctot = ctot/(len(self.x_train))
      cstr = cstr[:-1] + f") = {ctot:0.2f}"
      ## todo.. figure out how to get this textbox to extend to the width of the subplot
      c = self.ax.text(0.05,0.02,cstr, transform=self.ax.transAxes, color=dlc["dlpurple"])
      self.cost_items.append(c)


class contour_and_surface_plot:
  ''' plots combined in class as they have similar operations '''
  # pylint: disable=missing-function-docstring
  # pylint: disable=attribute-defined-outside-init
  def __init__(self, axc, axs, x_train, y_train, w_range, b_range, w, b):

      self.x_train = x_train
      self.y_train = y_train
      self.axc = axc
      self.axs = axs

      #setup useful ranges and common linspaces
      b_space  = np.linspace(*b_range, 100)
      w_space  = np.linspace(*w_range, 100)

      # get cost for w,b ranges for contour and 3D
      tmp_b,tmp_w = np.meshgrid(b_space,w_space)
      z = np.zeros_like(tmp_b)
      for i in range(tmp_w.shape[0]):
          for j in range(tmp_w.shape[1]):
              z[i, j] = compute_cost_matrix(x_train.reshape(-1,1), y_train, tmp_w[i,j], tmp_b[i,j],
                                            logistic=True, lambda_=0, safe=True)
              if z[i,j] == 0:
                  z[i,j] = 1e-9

      ### plot contour ###
      CS = axc.contour(tmp_w, tmp_b, np.log(z), levels=12, linewidths=2, alpha=0.7, cmap=cm.coolwarm)
      axc.set_title('log(Cost(w,b))')
      axc.set_xlabel('w', fontsize=10)
      axc.set_ylabel('b', fontsize=10)
      axc.set_xlim(w_range)
      axc.set_ylim(b_range)
      self.update_contour_wb_lines(w, b, firsttime=True)
      axc.text(0.7,0.05,"Click to choose w, b", bbox=dict(facecolor='white', ec='black'), fontsize = 10,
              transform=axc.transAxes, verticalalignment = 'center', horizontalalignment='center')
      axc.set_aspect("equal")        

      #Surface plot of the cost function J(w,b)
      axs.plot_surface(tmp_w, tmp_b, z, cmap=cm.coolwarm, alpha=0.3, antialiased=True)
      axs.plot_wireframe(tmp_w, tmp_b, z, color='k', alpha=0.1)
      axs.set_xlabel("$w$")
      axs.set_ylabel("$b$")
      axs.zaxis.set_rotate_label(False)
      axs.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
      axs.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
      axs.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
      axs.set_zlabel("J(w, b)", rotation=90)
      axs.view_init(30, -120)

      axs.autoscale(enable=False)
      axc.autoscale(enable=False)

      self.path = path(self.w, self.b, self.axc)  # initialize an empty path, avoids existance check

  def update_contour_wb_lines(self, w, b, firsttime=False):
      self.w = w
      self.b = b
      cst = compute_cost_matrix(self.x_train.reshape(-1,1), self.y_train, np.array(self.w), self.b,
                                logistic=True, lambda_=0, safe=True)

      # remove lines and re-add on contour plot and 3d plot
      if not firsttime:
          for artist in self.dyn_items:
              artist.remove()
      a = self.axc.scatter(self.w, self.b, s=100, color=dlc["dlblue"], zorder=10, label="cost with \ncurrent w,b")
      b = self.axc.hlines(self.b, self.axc.get_xlim()[0], self.w, lw=2, alpha=0.3, color='k', ls='dashed')
      c = self.axc.vlines(self.w, self.axc.get_ylim()[0], self.b, lw=2, alpha=0.3, color='k', ls='dashed')
      d = self.axc.annotate(f"Cost: {cst:0.2f}", xy= (self.w, self.b), xytext = (4,4), textcoords = 'offset points',
                          bbox=dict(facecolor='white'), size = 10)
      #Add point in 3D surface plot
      e = self.axs.scatter3D(self.w, self.b, cst , marker='X', s=100)

      self.dyn_items = [a,b,c,d,e]


class cost_plot:
  """ manages cost plot for plt_quad_logistic """
  # pylint: disable=missing-function-docstring
  # pylint: disable=attribute-defined-outside-init
  def __init__(self,ax):
      self.ax = ax
      self.ax.set_ylabel("log(cost)")
      self.ax.set_xlabel("iteration")
      self.costs = []
      self.cline = self.ax.plot(0, 0, color=dlc["dlblue"])

  def re_init(self):
      self.ax.clear()
      self.__init__(self.ax)

  def add_cost(self,J_hist):
      self.costs.extend(J_hist)
      self.cline[0].remove()
      self.cline = self.ax.plot(self.costs)

class path:
  ''' tracks paths during gradient descent on contour plot '''
  # pylint: disable=missing-function-docstring
  # pylint: disable=attribute-defined-outside-init
  def __init__(self, w, b, ax):
      ''' w, b at start of path '''
      self.path_items = []
      self.w = w
      self.b = b
      self.ax = ax

  def re_init(self, w, b):
      for artist in self.path_items:
          artist.remove()
      self.path_items = []
      self.w = w
      self.b = b

  def add_path_item(self, w, b):
      a = FancyArrowPatch(
          posA=(self.w, self.b), posB=(w, b), color=dlc["dlblue"],
          arrowstyle='simple, head_width=5, head_length=10, tail_width=0.0',
      )
      self.ax.add_artist(a)
      self.path_items.append(a)
      self.w = w
      self.b = b   