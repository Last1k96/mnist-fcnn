# MNIST-FCNN
������������ ������, ���� ������� - ������� ����� ��������� ��������������� ������ �� ������� ������ ������������� ���� �� ��������� ������ [MNIST](http://yann.lecun.com/exdb/mnist/ "MNIST dataset") � �������������� ������������ ��������� ���� � ����� ������� �����. 

## �������� ����
- �� ���� �������� ������������� ����������� ����� 28�28 ��������.
- ���������� �������� �� ������� ���� �������� ���������� ��������� ������.
- �� �������� ���� 10 ��������.
- � �������� ������� ��������� �� ������� ���� ������������ ��������������� �������, � �� �������� - softmax.
- ������� ������ - cross-entropy.

## ������
������ ��������� � ����� *theory*.

## ����������� �������
- ��������� ��������� ISO C++17.
- OpenMP

## ������
� ����� � �������� ��������� ������ *mnist.py*, ������� �������� ������������� � �������� ������ � ����� *mnist*. ������ ��������� ������������ �� ���������� �����������:

```
mnist-fcnn [���� �� MNIST] [���������� ����] [�������� ��������] [���-�� �������� �� ������� ����] [������ ������ "������"] [���-�� ��-�� ��������� �������]
```
��������� �� ���������:
```
mnist-fcnn mnist 10 0.2 100 100 60000
```
������ �������� ���� �� 120 ��������� �� ������� ���� � ������� 10 ���� �� ��������� 0.1 ������� �� 30 ����������� �� ��� ������������� ��������� (60000 ���������):
```
mnist-fcnn "D:\dev\mnist" 10 0.1 120 30
```

## ������������
- Windows 10 x64
- ��������� Intel Core i5-6600K, 4 ����, ����������� �� 4.3���
- ����������� ������ 2x DIMM DDR4, 8��, 2133���

���������� ������ ��������� � ����� *test.txt*

����������� ��������� �������� ����:
```
Epoch count: 20
Learning rate: 0.3
Hidden layer size: 100
Batch size: 30
```
�� 9-�� ����� ������ ���� = 0.1030, � �������� = 0.9711.
P.S. ������ ������������� ����������� ����������� ������ � �������������� ���������������.


