NVCC       = nvcc  
INCLUDES   = -I./inc  
LIBS       = -lcublas -lcusolver -lcurand  

all:  mydoubleblocking_Iter_V6  


mydoubleblocking_Iter_V6: src/mydoubleblocking_Iter_V6.cu src/myutils.cu  
	$(NVCC) $(INCLUDES) src/mydoubleblocking_Iter_V6.cu src/myutils.cu $(LIBS) -o $@  

clean:  
	rm -f  mydoubleblocking_Iter_V6  