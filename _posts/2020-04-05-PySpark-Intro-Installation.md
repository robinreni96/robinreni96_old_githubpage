---
layout: single
header:
  teaser: /assets/images/pyspark.png
title: "PySpark Intro & Installation - Ubuntu 18.04, Python 3.6"
date: 2020-04-05 20:00:00 -0800
categories: Data-Mining
tags:
  - Distributed Data Processing
  - AI
  - Python
---

![PySpark Logo](https://databricks.com/wp-content/uploads/2018/12/PySpark-1024x164.png)

**Apache Spark** is an open source analytics engine used for big data workloads. It can handle both batches as well as real-time analytics and data processing workloads. It is based on **Hadoop MapReduce** and it extends the **MapReduce** model to efficiently use it for more types of computations, which includes interactive queries and stream processing. Spark provides native bindings for the **Java, Scala, Python, and R** programming languages.

In collaboration with Apache Spark , Python supports Spark API's to take advantage of distributed real time data processing. PySpark, helps you interface with Resilient Distributed Datasets  in Apache Spark and Python programming language. This has been achieved by taking advantage of the Py4j library. **Py4J** is a popular library which is integrated within PySpark and allows python to dynamically interface with JVM objects. Thats how Pyspark got JVM support for processing.

### Installation of PySpark :
### 1) Java 8 Installation:
If you already have java 8 installed and default please skip this , otherwise
```
$ sudo apt install openjdk-8-jdk
```
If you have already Java 11, you need to change the default to Java 8 . Because PySpark widely supports Java8 . To set java8 default:
```
$ sudo update-alternatives --config java
```
Select Java 8 and then confirm your changes. To check the version
```
$ java -version
```
Output should be:
```
openjdk version "1.8.0_242"
OpenJDK Runtime Environment (build 1.8.0_242-8u242-b08-0ubuntu3~18.04-b08)
OpenJDK 64-Bit Server VM (build 25.242-b08, mixed mode)
```
Now you successfully installed Java 8 for Pyspark.

### 2) Python 3.6+ Installation:
If you already installed python 3.6+ please skip to next step , Otherwise:
```
$ sudo apt install python3
```
To check python ,
```
$ python3
```
Output should be:
```
Python 3.6.9 (default, Nov  7 2019, 10:44:02)
[GCC 8.3.0] on linux
Type "help", "copyright", "credits" or "license" for more information.
>>>
```
### 3) Download Spark File and Save it in $HOME:
**Download spark from** [**https://spark.apache.org/downloads.html**](https://spark.apache.org/downloads.html?)
![spark_download](https://lh3.googleusercontent.com/4Xnh_WLSIOdmOvC77fVq8z_KnxkrhEcPXkSpADNGx-4R8LLrE9M5iA-zw0q1IzBsMPxyH34ZCMb9 "spark")

After downloaded, extract the file & move to home. Follow the command
```
$ cd ~
$ cd Downloads
$ tar -zxvf spark-2.4.5-bin-hadoop2.7.tgz
$ cd ~
$ mkdir spark
$ cd spark
$ sudo mv Downloads/spark-2.4.5-bin-hadoop2.7 .
```
Now you successfully moved the source file of PySpark to home directory

### 4) Set the $JAVA_HOME environment variable:
For this, run the following in the terminal:
```
$ sudo vim /etc/environment
```
It will open the file in vim. Press **i** to insert.  Then, in a new line after the PATH variable add
```
JAVA_HOME="/usr/lib/jvm/java-8-openjdk-amd64"
```
then press **"Esc -> : -> w -> q -> Enter"** . It is to save the update in the file.
Then in the terminal :
```
$ source /etc/environment
```
Now if you run :
```
$ echo $JAVA_HOME
```
You can see the output:
```
/usr/lib/jvm/java-8-openjdk-amd64
```

### 5) Configure environment variables for spark:
Now we have to set the environment variables of PySpark to your linux system
```
$ vim ~/.bashrc
```
Press **i** to insert . Paste the following variables to end of the file
```
source /etc/environment
export SPARK_HOME=~/spark/spark-2.4.5-bin-hadoop2.7/
export PATH=$PATH:$SPARK_HOME/bin
export PYTHONPATH=$SPARK_HOME/python:$PYTHONPATH
export PYSPARK_PYTHON=/usr/bin/python3
export PATH=$PATH:$JAVA_HOME/jre/bin
```
Save the file and exit by pressing **"Esc -> : -> w -> q -> Enter"** . Finally, load the .bashrc file again in the terminal by
```
$ source ~/.bashrc
```
All setup done. Time to check  !

### 6) Run PySpark
Open a new terminal , Run
```
$ pyspark
```
You should see a output like this
```
Welcome to
      ____              __
     / __/__  ___ _____/ /__
    _\ \/ _ \/ _ `/ __/  '_/
   /__ / .__/\_,_/_/ /_/\_\   version 2.4.5
      /_/

Using Python version 3.6.9 (default, Nov  7 2019 10:44:02)
>>>
```

Hurrah ! You successfully installed and configured PySpark . Now you are ready to go .

### Advantages of using PySpark:
• Python is very easy to learn and implement.  
• It provides simple and comprehensive API.  
• With Python, the readability of code, maintenance, and familiarity is far better.  
• It features various options for data visualization, which is difficult using Scala or Java.

### Reference Links :
* [https://towardsdatascience.com/installing-pyspark-with-java-8-on-ubuntu-18-04-6a9dea915b5b](https://towardsdatascience.com/installing-pyspark-with-java-8-on-ubuntu-18-04-6a9dea915b5b)
* [https://medium.com/@GalarnykMichael/install-spark-on-ubuntu-pyspark-231c45677de0](https://medium.com/@GalarnykMichael/install-spark-on-ubuntu-pyspark-231c45677de0)
