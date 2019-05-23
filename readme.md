生成对抗网络GAN损失函数Loss的计算：
辨别器对假数据的损失原理相同，最终达到的目标是对于所有的真实图片，输出为1；对于所有的假图片，输出为0。
生成器的目标是愚弄辨别器蒙混过关，需要达到的目标是对于生成的图片，输出为1(正好和鉴别器相反).
code:

d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_real,labels=tf.ones_like(d_logits_real) * 
(1 - smooth))) 

d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_fake,labels=tf.zeros_like(d_logits_real)))

d_loss = d_loss_real + d_loss_fake 

g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_fake,labels=tf.ones_like(d_logits_fake))


lan Goodfellow:
当我开始第一次尝试构建 GAN 时，那时有非常多的优秀工具，如 Theano、LISA lab 计算机集群等。所以编写 GAN 不那么困难的原因有一部分就是因为这些优秀的深度学习框架，同时我在整个博士期间都在学习深度学习，因此有非常多的代码块可随时嵌入到新模型中。我第一个 GAN 的实现主要是从 MNIST 分类器代码中复制粘贴
