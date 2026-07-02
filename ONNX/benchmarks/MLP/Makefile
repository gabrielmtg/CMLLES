.PHONY: all clean train train_iris train_iaes

all:
	$(MAKE) -C Iris all
	$(MAKE) -C IAES all

train: train_iris train_iaes

train_iris:
	$(MAKE) -C Iris train

train_iaes:
	$(MAKE) -C IAES train

clean:
	$(MAKE) -C Iris clean
	$(MAKE) -C IAES clean
