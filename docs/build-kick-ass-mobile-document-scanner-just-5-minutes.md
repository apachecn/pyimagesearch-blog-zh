# 如何在 5 分钟内建立一个强大的移动文档扫描仪

> 原文：<https://pyimagesearch.com/2014/09/01/build-kick-ass-mobile-document-scanner-just-5-minutes/>

[![](img/f4893d6a2865f6495326499693e9bfa4.png)](https://pyimagesearch.com/wp-content/uploads/2014/08/receipt-scanned.jpg)

用 OpenCV 构建一个文档扫描仪只需要三个简单的步骤:

*   **第一步:**检测边缘。
*   **第二步:**利用图像中的边缘找到代表被扫描纸张的轮廓(轮廓)。
*   **步骤 3:** 应用透视变换获得文档的自顶向下视图。

真的。就是这样。

只需三个步骤，您就可以向 App Store 提交自己的文档扫描应用程序了。

听起来有趣吗？

请继续阅读。并解开秘密，建立一个自己的移动扫描仪应用程序。

**OpenCV 和 Python 版本:**
这个例子将在 **Python 2.7/3+** 和 **OpenCV 2.4/3+** 上运行

# 如何在 5 分钟内建立一个强大的移动文档扫描仪

<https://www.youtube.com/embed/yRer1GC2298?feature=oembed>