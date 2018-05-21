# 第11周“图像SIFT特征”的课后作业如下：

### 理解和实践SIFT特征提取与匹配。

自己拍摄两张有部分区域重叠的照片，使用opencv的特征提取、匹配函数实现两张图片的SIFT特征提取和匹配，并绘制特征点在匹配前和匹配后的连线。
类似网址 https://blog.csdn.net/dcrmg/article/details/52578732 的试验效果。



由于在学习图像识别中的特征点检测中，需要用到Surf和Sift算法，但是这两个算法在OpenCV 3.1.0的Release版本中并不存在，因为他们是存放在opencv_contrib目录下面的未稳定功能模块。

首先查看我的OpenCV版本`std::cout << "opencv version:" << CV_VERSION << std::endl;`

![image-20180521193202588](/var/folders/pz/8hdd81_959q73c1y_m3gpm6c0000gn/T/abnerworks.Typora/image-20180521193202588.png)

很不幸是3.1.0。所以如果想要使用这个目录的功能，就需要自己重新进行OpenCV的编译。

根据opencv_contrib的Github repohttps://github.com/Itseez/opencv_contrib

参照README进行操作。由于我的opencv是通过conda下载的，未找到本地的build和sources目录。也不太想浪费太多时间在配置环境上面。故此实验在windows虚拟机opencv_2.4.13完成。

- 代码

```c++
#include <opencv2/core/core.hpp>  
#include <opencv2/highgui/highgui.hpp>  
#include <opencv2/nonfree/features2d.hpp>  
#include <iostream>  

using namespace std;
using namespace cv;

int main()
{
	//读取图片文件
	Mat img1 = imread("3321.jpg");
	Mat img2 = imread("321.jpg");
	//如果读取失败退出
	if (!img1.data || !img2.data)
	{
		cout << "Error reading images!!" << endl;
		return -1;
	}
	//设置SIFT找出img1和img2的特征点
	SiftFeatureDetector siftDetector;
	vector<KeyPoint> keypoints1, keypoints2;
	siftDetector.detect(img1, keypoints1);
	siftDetector.detect(img2, keypoints2);
	cout << "Number of detected keypoints:\nimg1:" << keypoints1.size() << "points.\nimg2:"
		<< keypoints2.size() << "points." << endl;

	SiftDescriptorExtractor siftExtractor;
	Mat descriptor1, descriptor2;
	siftExtractor.compute(img1, keypoints1, descriptor1);
	siftExtractor.compute(img2, keypoints2, descriptor2);
	cout << "Number of Descriptors1:" << descriptor1.rows << endl;
	cout << "Number of Descriptors2:" << descriptor2.rows << endl;
	cout << "Demension of sift Descriptors:" << descriptor1.cols << endl;

	//画出img1和img2的特征点
	Mat imgkey1, imgkey2;
	drawKeypoints(img1, keypoints1, imgkey1, Scalar::all(-1));
	drawKeypoints(img2, keypoints2, imgkey2, Scalar::all(-1));
	imshow("3321", imgkey1);
	imshow("321", imgkey2);

	//关联img1和img2的特征点
	vector<DMatch> matches;
	FlannBasedMatcher siftMatcher;
	siftMatcher.match(descriptor1, descriptor2, matches, Mat());

	//寻找匹配精度小于最小距离两倍的匹配点集
	double dist_max = 0;
	double dist_min = 100;
	for (int i = 0; i < descriptor1.rows; i++)
	{
		double dist = matches[i].distance;
		if (dist < dist_min)
			dist_min = dist;
		if (dist > dist_max)
			dist_max = dist;
	}

	vector<DMatch> goodmatches;
	for (int i = 0;i < matches.size(); i++)
	{
		if (matches[i].distance < 2 * dist_min)
			goodmatches.push_back(matches[i]);
	}

	//画出关联的图片
	Mat imgmatches;
	drawMatches(img1,
		keypoints1,
		img2,
		keypoints2,
		goodmatches,
		imgmatches,
		Scalar::all(-1),
		Scalar::all(-1),
		vector<char>(),
		DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
	imshow("Match Results:", imgmatches);
	imwrite("3221.jpg", imgmatches);
	waitKey(0);
	return 0;
}
```

- 结果

<img src="https://ws4.sinaimg.cn/large/006tKfTcgy1frj9201knwj30ip0c7wer.jpg" width="600px">

<img src="https://ws3.sinaimg.cn/large/006tKfTcgy1frj7bzk7upj31120rtqe7.jpg" width="400px">

<img src="https://ws3.sinaimg.cn/large/006tKfTcgy1frj7dmc5ytj31120rt18q.jpg" width="400px">

<img src="https://ws2.sinaimg.cn/large/006tKfTcgy1frj7efud5gj31kw0lcaxj.jpg" width="600px">



### 全景图片的生成练习。

旋转相机拍摄多张室内或室外图片，然后利用opencv的相关函数完成一幅全景图片的制作，并解释全景图片的实现过程。
类似效果如网址 https://blog.csdn.net/dcrmg/article/details/52653366 。

- 实现过程

(1)图像采集

(2)图像预处理，进行图像校正和图像去燥等预处理，减少图像拼接时的干扰因素

(3)提取待拼接图像的特征信息，并对这些信息进行分类，筛选，用得到的特征信息对图像进行特征匹配

(4)对已匹配的图像融合处理，消除图像匹配时留下的痕迹

(5)得到目标图像

- 代码

```c++
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/stitching/stitcher.hpp>
using namespace std;
using namespace cv;

bool try_use_gpu = false;
vector<Mat> imgs;

int main()
{
	Mat img1 = imread("1.jpg");
	Mat img2 = imread("2.jpg");
	Mat img3 = imread("3.jpg");
	Mat img4 = imread("4.jpg");
	if (img1.empty() || img2.empty())
	{
		cout << "Can't read image" << endl;
		return -1;
	}
	imgs.push_back(img1);
	imgs.push_back(img2);
	imgs.push_back(img3);
	imgs.push_back(img4);
	Stitcher stitcher = Stitcher::createDefault(try_use_gpu);
	// 使用stitch函数进行拼接
	Mat pano;
	Stitcher::Status status = stitcher.stitch(imgs, pano);
	imwrite("stitchingImg.jpg", pano);
	Mat pano2 = pano.clone();
	// 显示源图像，和结果图像
	imshow("stitchingImg", pano);
	if (waitKey() == 27)
		return 0;
}
```



- 结果

<img src="https://ws1.sinaimg.cn/large/006tKfTcgy1frj7k0l1grj31120rt405.jpg" width="400px">

<img src="https://ws4.sinaimg.cn/large/006tKfTcgy1frj7kjcl2jj31120rtgnk.jpg" width="400px">

<img src="https://ws3.sinaimg.cn/large/006tKfTcgy1frj7lcyaqzj31120rtgnv.jpg" width="400px">

<img src="https://ws2.sinaimg.cn/large/006tKfTcgy1frj7lzli5zj31120rtgns.jpg" width="400px">

<img src="https://ws3.sinaimg.cn/large/006tKfTcgy1frj7mp6xkwj31kw0ip11k.jpg" width="700px">

### 心得体会

SIFT算法，即尺度不变特征变换（Scale-invariant feature transform，SIFT），是用于图像处理领域的一种描述。这种描述具有尺度不变性，可在图像中检测出关键点，是一种局部特征描述子。 该方法于1999年由David Lowe  首先发表于计算机视觉国际会议（International Conference on Computer Vision，ICCV），2004年再次经David Lowe整理完善后发表于International journal of computer vision（IJCV） 。截止2014年8月，该论文单篇被引次数达25000余次。

通过课上学习，了解了图像拼接技术和SIFT特征点提取的过程和特点，课下通过强大的opencv库调用相关函数进行处理，效果还是比较理想的。