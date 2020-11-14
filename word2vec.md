### OneHot
Ex.  
["male", "female"]  
["from Europe", "from US", "from Asia"]  
["uses Firefox", "uses Chrome", "uses Safari", "uses Internet Explorer"]   
&ensp;---->  
feature1=[01,10]  
feature2=[001,010,100]  
feature3=[0001,0010,0100,1000]    
 
### Word Embedding
将高维词向量嵌入到一个低维空间。如下图我们将词汇表里的词用"Royalty","Masculinity", "Femininity"和"Age"4个维度来表示，King这个词对应的词向量可能是(0.99,0.99,0.05,0.7)  
![avatar](https://upload-images.jianshu.io/upload_images/9285151-23ea3a6a14783ea9.png?imageMogr2/auto-orient/strip|imageView2/2/w/600/format/webp)
![avatar](https://upload-images.jianshu.io/upload_images/9285151-548de7208a382a0c.png?imageMogr2/auto-orient/strip|imageView2/2/w/468/format/webp)  

### Word2Vec
word2vec模型其实就是简单化的神经网络  
![avatar](https://upload-images.jianshu.io/upload_images/9285151-c719e0fee3d2bcb6.png?imageMogr2/auto-orient/strip|imageView2/2/w/670/format/webp)  
输入是One-Hot Vector，Hidden Layer没有激活函数，也就是线性的单元。Output Layer维度跟Input Layer的维度一样，用的是Softmax回归。当这个模型训练好以后，我们并不会用这个训练好的模型处理新的任务，我们真正需要的是这个模型通过训练数据所学得的参数，例如隐层的权重矩阵。  
根据输入输出的不同可以分为两种模型：CBOW(Continuous Bag-of-Words)与Skip-Gram。CBOW模型的训练输入是某一个特征词的上下文相关的词对应的词向量，而输出就是这特定的一个词的词向量。　Skip-Gram模型和CBOW的思路是反着来的，即输入是特定的一个词的词向量，而输出是特定词对应的上下文词向量。CBOW对小型数据库比较合适，而Skip-Gram在大型语料中表现更好。  

#### CBOW
![avatar](https://upload-images.jianshu.io/upload_images/9285151-bf3d31fd22025027.png?imageMogr2/auto-orient/strip|imageView2/2/w/421/format/webp)  
1 输入层：上下文单词的onehot. {假设单词向量空间dim为V，上下文单词个数为C}  
2 所有onehot分别乘以共享的输入权重矩阵W. {VN矩阵，N为自己设定的数，初始化权重矩阵W}  
3 所得的向量 {因为是onehot所以为向量} 相加求平均作为隐层向量, size为1N.  
4 乘以输出权重矩阵W' {NV}  
5 得到向量 {1V} 激活函数处理得到V-dim概率分布 {PS: 因为是onehot嘛，其中的每一维斗代表着一个单词}  
6 概率最大的index所指示的单词为预测出的中间词（target word）与true label的onehot做比较，误差越小越好（根据误差更新权重矩阵）  
所以，需要定义loss function（一般为交叉熵代价函数），采用梯度下降算法更新W和W'。训练完毕后，输入层的每个单词与矩阵W相乘得到的向量的就是我们想要的词向量（word embedding），这个矩阵（所有单词的word embedding）也叫做look up table（其实聪明的你已经看出来了，其实这个look up table就是矩阵W自身），也就是说，任何一个单词的onehot乘以这个矩阵都将得到自己的词向量。有了look up table就可以免去训练过程直接查表得到单词的词向量了。  
![avatar](https://upload-images.jianshu.io/upload_images/9285151-21157b859f9f3b2b.png?imageMogr2/auto-orient/strip|imageView2/2/w/518/format/webp)    
Ex.  
![avatar](https://upload-images.jianshu.io/upload_images/9285151-e293f7bb529f5e55.png?imageMogr2/auto-orient/strip|imageView2/2/w/720/format/webp)
![avatar](https://upload-images.jianshu.io/upload_images/9285151-a119b8c935b22164.png?imageMogr2/auto-orient/strip|imageView2/2/w/720/format/webp)  
窗口大小是2，表示选取coffe前面两个单词和后面两个单词，作为input词。  
假设我们此时得到的概率分布已经达到了设定的迭代次数，那么现在我们训练出来的look up table应该为矩阵W。即，任何一个单词的one-hot表示乘以这个矩阵都将得到自己的word embedding。  

#### Skip-Gram
![avatar](https://upload-images.jianshu.io/upload_images/9285151-fca1fdda41d6d422.png?imageMogr2/auto-orient/strip|imageView2/2/w/462/format/webp)  
Ex.  
假如我们有一个句子“The dog barked at the mailman”。  
首先我们选句子中间的一个词作为我们的输入词，例如我们选取“dog”作为input word；  
有了input word以后，我们再定义一个叫做skip_window的参数，它代表着我们从当前input word的一侧（左边或右边）选取词的数量。如果我们设置skip_window=2，那么我们最终获得窗口中的词（包括input word在内）就是['The', 'dog'，'barked', 'at']。skip_window=2代表着选取左input word左侧2个词和右侧2个词进入我们的窗口，所以整个窗口大小span=2x2=4。另一个参数叫num_skips，它代表着我们从整个窗口中选取多少个不同的词作为我们的output word，当skip_window=2，num_skips=2时，我们将会得到两组 (input word, output word) 形式的训练数据，即 ('dog', 'barked')，('dog', 'the')。  
神经网络基于这些训练数据将会输出一个概率分布，这个概率代表着我们的词典中的每个词是output word的可能性。这句话有点绕，我们来看个栗子。第二步中我们在设置skip_window和num_skips=2的情况下获得了两组训练数据。假如我们先拿一组数据 ('dog', 'barked') 来训练神经网络，那么模型通过学习这个训练样本，会告诉我们词汇表中每个单词是“barked”的概率大小。  
模型的输出概率代表着到我们词典中每个词有多大可能性跟input word同时出现。举个栗子，如果我们向神经网络模型中输入一个单词“中国“，那么最终模型的输出概率中，像“英国”， ”俄罗斯“这种相关词的概率将远高于像”苹果“，”蝈蝈“非相关词的概率。因为”英国“，”俄罗斯“在文本中更大可能在”中国“的窗口中出现。我们将通过给神经网络输入文本中成对的单词来训练它完成上面所说的概率计算。  
下面的图中给出了一些我们的训练样本的例子。我们选定句子“The quick brown fox jumps over lazy dog”，设定我们的窗口大小为2（window_size=2），也就是说我们仅选输入词前后各两个词和输入词进行组合。下图中，蓝色代表input word，方框内代表位于窗口内的单词。Training Samples（输入， 输出）  
![avatar](https://upload-images.jianshu.io/upload_images/9285151-cc3ff3bae4ec0be4.png?imageMogr2/auto-orient/strip|imageView2/2/w/600/format/webp)  
我们的模型将会从每对单词出现的次数中习得统计结果。例如，我们的神经网络可能会得到更多类似（“中国“，”英国“）这样的训练样本对，而对于（”英国“，”蝈蝈“）这样的组合却看到的很少。因此，当我们的模型完成训练后，给定一个单词”中国“作为输入，输出的结果中”英国“或者”俄罗斯“要比”蝈蝈“被赋予更高的概率。  
![avatar](https://img-blog.csdnimg.cn/20190518164912509.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80MTg0MzkxOA==,size_16,color_FFFFFF,t_70)  

### Reference
https://www.jianshu.com/p/471d9bfbd72f  
https://blog.csdn.net/weixin_41843918/article/details/90312339
