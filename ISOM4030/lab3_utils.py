import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.gridspec import GridSpec
from matplotlib import cm
import numpy as np

dlblue = '#0096ff'; dlorange = '#FF9300'; dldarkred='#C00000'; dlmagenta='#FF40FF'; dlpurple='#7030A0'
dlcolors = [dlblue, dlorange, dldarkred, dlmagenta, dlpurple]


n_bin = 5
dlcm = LinearSegmentedColormap.from_list('dl_map', dlcolors, N=n_bin)


def mk_cost_lines(x, y, w, b, ax):
    ''' makes vertical cost lines'''
    cstr = "cost = (1/m)*("
    ctot = 0
    label = 'cost for point'
    ax.vlines(x, b + w * x, y, alpha=0.3, color='k', label=label)  # lw=3, ls='dotted'   # draw vertical bars denoting residuals
    c_p = ((b + w * x - y)**2) / 2
    label=''    # just one
    for v, label_x, label_y in zip(c_p, x, (y + b + w * x)/2):
      ax.annotate(f'{v:0.0f}', xy=(label_x, label_y), xycoords='data', color=dlpurple, xytext=(5, 0), textcoords='offset points')
      cstr += f"{v:0.0f} +"
      ctot += v  
    ctot = ctot/(len(x))
    cstr = cstr[:-1] + f") = {ctot:0.0f}"
    ax.text(0.15,0.02,cstr, transform=ax.transAxes, color=dlpurple) 


def plt_house_x(X, y, f_wb=None, ax=None):
    ''' plot house with aXis '''
    if not ax:
        fig, ax = plt.subplots(1,1)
    ax.scatter(X, y, marker='x', c='r', label="Actual Value")
    ax.set_title("Housing Prices")
    ax.set(xlabel='Size (1000 sqft)', ylabel='Price (in 1000s of dollars)')
    if f_wb is not None:
        ax.plot(X, f_wb,  c=dlblue, label="Our Prediction")
    ax.legend()


# this is the 2D cost curve with interactive slider

def plt_stationary(x_train, y_train):
    # setup figure
    fig = plt.figure(figsize=(12, 10))
    # fig = plt.figure(constrained_layout=True,  figsize=(12,10))
    fig.set_facecolor('#ffffff') #white
    fig.canvas.toolbar_position = 'top'
    # gs = GridSpec(5, 2, figure=fig, wspace = 0.01)
    gs = GridSpec(5, 2, figure=fig)
    ax0 = fig.add_subplot(gs[:2, 0])
    ax1 = fig.add_subplot(gs[:2, 1])
    ax2 = fig.add_subplot(gs[2:, :], projection='3d')
    ax = np.array([ax0, ax1, ax2])

    # setup useful ranges and common linspaces
    w_range = np.array([200-300.,200+300])
    b_range = np.array([50-300., 50+300])
    b_space  = np.linspace(*b_range, 100)
    w_space  = np.linspace(*w_range, 100)

    # get cost for w, b ranges for contour and 3D
    tmp_b, tmp_w = np.meshgrid(b_space, w_space)
    z = np.zeros_like(tmp_b)
    
    for (i, j), v in np.ndenumerate(z):
    # compute the mean squrared error cost
       z[i, j] = ((y_train - tmp_b[i, j] - tmp_w[i, j] * x_train)**2).mean()/ 2
       if z[i, j] == 0: z[i,j] = 1e-6
    
    w0 = 200; b = -100    #initial point
    ### plot model w cost ###
    f_wb = x_train * w0 + b
    mk_cost_lines(x_train, y_train, w0, b, ax[0])
    plt_house_x(x_train, y_train, f_wb=f_wb, ax=ax[0])

    ### plot contour ###
    CS = ax[1].contour(tmp_w, tmp_b, np.log(z), levels=12, linewidths=2, alpha=0.7, cmap=cm.coolwarm)  # colors=dlcolors
    ax[1].set_title('Cost(w,b)')
    ax[1].set_xlabel('w', fontsize=10)
    ax[1].set_ylabel('b', fontsize=10)
    ax[1].set(xlim=w_range, ylim=b_range) 
    cscat  = ax[1].scatter(w0, b, s=100, color=dlblue, zorder=10, label="cost with \ncurrent w,b")
    chline = ax[1].hlines(b, ax[1].get_xlim()[0], w0, lw=2, alpha=0.3, color='k', ls='dashed')  # color=dlpurple, 
    cvline = ax[1].vlines(w0, ax[1].get_ylim()[0], b, lw=2, alpha=0.3, color='k', ls='dashed')
    ax[1].text(0.5,0.95,"Click to choose w,b",  bbox=dict(facecolor='white', ec = 'black'), fontsize = 10,
                transform=ax[1].transAxes, verticalalignment = 'center', horizontalalignment= 'center')
    ax[1].set_aspect("equal")               

    # Surface plot of the cost function J(w,b)
    ax[2].plot_surface(tmp_w, tmp_b, z, cmap = cm.coolwarm, alpha=0.3, antialiased=True)  #cmap = dlcm
    ax[2].plot_wireframe(tmp_w, tmp_b, z, color='k', alpha=0.1)
    plt.xlabel("$w$")
    plt.ylabel("$b$")
    ax[2].zaxis.set_rotate_label(False)
    ax[2].xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax[2].yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax[2].zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax[2].set_zlabel("J(w, b)\n\n", rotation=90)
    plt.title("Cost(w,b) \n [You can rotate this figure]", size=12)
    ax[2].view_init(40, -120)
    plt.tight_layout()

    return fig,ax, [cscat, chline, cvline]

# https://matplotlib.org/stable/users/event_handling.html

class plt_update_onclick:
    def __init__(self, fig, ax, x_train,y_train, dyn_items):
        self.fig = fig
        self.ax = ax
        self.x_train = x_train
        self.y_train = y_train
        self.dyn_items = dyn_items
        self.cid = fig.canvas.mpl_connect('button_press_event', self)

    def __call__(self, event):
        if event.inaxes == self.ax[1]:
            ws = event.xdata
            bs = event.ydata
            # compute the mean squrared error cost
            cst = ((self.y_train - bs - ws * self.x_train)**2).mean()/ 2

            # clear and redraw line plot
            self.ax[0].clear()
            f_wb = np.dot(self.x_train,ws) + bs
            mk_cost_lines(self.x_train, self.y_train,ws, bs, self.ax[0])
            plt_house_x(self.x_train, self.y_train, f_wb=f_wb, ax=self.ax[0])

            # remove lines and re-add on countour plot and 3d plot
            for artist in self.dyn_items:
                artist.remove()

            a = self.ax[1].scatter(ws,bs, s=100, color=dlblue, zorder= 10, label="cost with \ncurrent w,b")
            b = self.ax[1].hlines(bs, self.ax[1].get_xlim()[0], ws, lw=2, alpha=0.3, color='k', ls='dashed')
            c = self.ax[1].vlines(ws, self.ax[1].get_ylim()[0], bs, lw=2, alpha=0.3, color='k', ls='dashed')
            d = self.ax[1].annotate(f"Cost: {cst:.0f}", xy= (ws, bs), xytext = (4,4), textcoords = 'offset points',
                               bbox=dict(facecolor='white'), size = 10)

            # add point in 3D surface plot
            e = self.ax[2].scatter3D(ws, bs, cst , marker='X', s=100)

            self.dyn_items = [a,b,c,d,e]
            self.fig.canvas.draw()

def plt_gradient_descent_path(fig, ax, x_train, y_train, weight_history, w_range, b_range):
  b_space  = np.linspace(*b_range, 100)
  w_space  = np.linspace(*w_range, 100)

  B, W = np.meshgrid(b_space, w_space)

  param = np.expand_dims(np.array([*zip(B.ravel(), W.ravel())]), axis=1)  
  x_aug = np.expand_dims(np.stack([np.ones_like(x_train), x_train], axis=1), axis=0) 
  J = np.mean((np.expand_dims(y_train, axis=0) - np.sum(x_aug * param, axis=2))**2, axis=1)/ 2
  J = J.reshape(B.shape)
  
  b, w = [*zip(*weight_history)]
  ax.contour(W, B, np.log(J), levels=12, linewidths=2, alpha=0.7, cmap=cm.coolwarm)
  ax.scatter(w, b, s = 50, c='r', zorder = 3, alpha=0.4)  # edgecolor = 'k',linewidth = 2*math.sqrt((1/(float(j) + 1)))
  ax.axvline(x=0, color='k', alpha=0.5)
  ax.axhline(y=0, color='k', alpha=0.5)

  # plot arrows
  
  diff_b, diff_w = np.diff(b), np.diff(w)
  arrow_length = np.linalg.norm([*zip(diff_b, diff_w)], axis=1)
  head_length = 0.2
  alpha = (head_length - 0.35)/arrow_length + 1

  for i in range(len(diff_b)):
    if arrow_length[i] > head_length:
      ax.arrow(w[i], b[i], diff_w[i]*alpha[i], diff_b[i]*alpha[i], head_width=0.2,
               head_length=head_length, fc='k', ec='k', alpha=0.4, linewidth=1.5, zorder=2, length_includes_head=True)

  ax.set(xlim=w_range, ylim=b_range)
  ax.set_xlabel('w', fontsize=10)
  ax.set_ylabel('b', fontsize=10)
  ax.set_aspect("equal")            