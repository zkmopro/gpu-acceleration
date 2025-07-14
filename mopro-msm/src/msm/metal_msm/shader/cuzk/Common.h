#ifndef Common_h
#define Common_h

#include <simd/simd.h>

typedef struct {
  matrix_float4x4 trueMatrix;
  matrix_float4x4 falseMatrix;
  matrix_float4x4 otherMatrix;
} Uniforms;

typedef struct {
  int width;
  int height;
} Params;

typedef enum {
  VertexBuffer = 0,
  ParamsBuffer = 2
} BufferIndices;


#endif /* Common_h */
