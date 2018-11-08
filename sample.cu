#include <stdio.h>

int reduceByHost(int *a, int n){
   int sum = 0;
    for(int i = 0; i < n; i++){
        sum += a[i];
    }
    return sum;
}

/*
a0 -- a1 -- a2 -- a3 -- a4 -- a5 -- a6 -- a7 :step0
-- b0 -------- b1 -------- b2 ---------b3 -- :step1
---------c0 -------- c1 ---------c2--------- :step2
---------------d0-----------d1-------------- :step3
----------------------e0-------------------- :step4

max blocksize -> kich thuoc toi da cua 1 block

__syncthread() : dong bo hoa tat ca thread trong 1 block



*/

__global__ void reduceByDevice(int *a, int n){

}

