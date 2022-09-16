# 嘗試撰寫一個可以傳入時間序列3d points資料的畫圖class
from xml.dom.expatbuilder import FragmentBuilder
from matplotlib import pyplot as plt
import numpy as np
import mpl_toolkits.mplot3d.axes3d as p3
from matplotlib import animation

class VizAnimation:
    def __init__(self, data, data2) -> None:
        self.data = data
        self.data2 = data2
        self.timeCount = 0

        fig = plt.figure()
        ax = p3.Axes3D(fig, auto_add_to_figure=False)
        fig.add_axes(ax)

        ax.set_title('For testing')
        ax.set_xlabel('x axis')
        ax.set_ylabel('y axis')
        ax.set_zlabel('z axis')
        # TODO: 這邊可以創建多組points分別代表不同類別的資料
        #       但是底下update就不知道要怎麼回傳多組points的資料
        # Sol: 額外再開一個animation.FuncAnimation()即可
        # e.g. animation's positions, mapped positions ...
        # self.points, = ax.plot(data[0][0], data[0][1], data[0][2])
        self.points, = ax.plot(
            [i[0] for i in self.data[0]], 
            [i[1] for i in self.data[0]], 
            [i[2] for i in self.data[0]], 
            '.'
        )
        self.points2, = ax.plot(
            [i[0] for i in self.data2[0]], 
            [i[1] for i in self.data2[0]], 
            [i[2] for i in self.data2[0]]
        )


        self.animation = animation.FuncAnimation(
            fig, self.update, fargs=(self.points,self.data), frames=len(self.data), interval=500, blit=True)
        self.animation2 = animation.FuncAnimation(
            fig, self.update, fargs=(self.points2,self.data2), frames=len(self.data2), interval=500, blit=True)
        
        self.paused = False
        # fig.canvas.mpl_connect('button_press_event', self.toggle_pause)
        fig.canvas.mpl_connect('key_press_event', self.toggle_key_pressed)
        
    def toggle_pause(self, *args, **kwargs):
        if self.paused:
            self.animation.resume()
        else:
            self.animation.pause()
        self.paused = not self.paused
    
    def toggle_key_pressed(self, event):
        if event.key == 'a':
            if self.paused:
                self.animation.resume()
            else:
                self.animation.pause()
            self.paused = not self.paused
        
    def update(self, frameNum, points, data):
        points.set_data(
            [i[0] for i in data[frameNum]], [i[1] for i in data[frameNum]]
        )
        points.set_3d_properties([i[2] for i in data[frameNum]], 'z')
        return (points,)

if __name__ == '__main__':
    # 第一個維度是time
    # 第二個維度是point/joint
    # 第三個維度是XYZ
    sourceData = [
        [[0, 0, 0], [0, 1, 1], [2, 2, 2]], 
        [[0, 1, 0], [1, 1, 1], [2, 2, 1]], 
        [[0, 2, 0], [2, 1, 1], [2, 2, 0]]
    ]
    sourceData2 = [
        [[0, 0, 0], [0.5, 0, 0], [0.5, 0.5, 0]], 
        [[0, 0, 0], [1, 0, 0], [1, 1, 0]], 
        [[0, 0, 0], [1.5, 0, 0], [1.5, 1.5, 0]]
    ]
    # print([i[0] for i in sourceData[0]])    # all three points' x axis values in time 0

    # exit()
    vizAnim = VizAnimation(sourceData, sourceData2)
    plt.show()