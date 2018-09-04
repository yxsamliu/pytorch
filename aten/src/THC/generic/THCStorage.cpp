#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "generic/THCStorage.cpp"
#else

#include <ATen/core/intrusive_ptr.h>

real* THCStorage_(data)(THCState *state, const THCStorage *self)
{
  return self->data<real>();
}

ptrdiff_t THCStorage_(size)(THCState *state, const THCStorage *self)
{
  return THStorage_size(self);
}

int THCStorage_(elementSize)(THCState *state)
{
  return sizeof(real);
}

void THCStorage_(set)(THCState *state, THCStorage *self, ptrdiff_t index, real value)
{
  THArgCheck((index >= 0) && (index < self->size()), 2, "index out of bounds");
  hipStream_t stream = THCState_getCurrentStream(state);
  THCudaCheck(hipMemcpyAsync(THCStorage_(data)(state, self) + index, &value, sizeof(real),
                              hipMemcpyHostToDevice,
                              stream));
  THCudaCheck(hipStreamSynchronize(stream));
}

real THCStorage_(get)(THCState *state, const THCStorage *self, ptrdiff_t index)
{
  THArgCheck((index >= 0) && (index < self->size()), 2, "index out of bounds");
  real value;
  hipStream_t stream = THCState_getCurrentStream(state);
  THCudaCheck(hipMemcpyAsync(&value, THCStorage_(data)(state, self) + index, sizeof(real),
                              hipMemcpyDeviceToHost, stream));
  THCudaCheck(hipStreamSynchronize(stream));
  return value;
}

THCStorage* THCStorage_(new)(THCState *state)
{
  THStorage* storage = c10::make_intrusive<at::StorageImpl>(
      at::CTypeToScalarType<real>::to(),
      0,
      state->hipDeviceAllocator,
      true).release();
  return storage;
}

THCStorage* THCStorage_(newWithSize)(THCState *state, ptrdiff_t size)
{
  THStorage* storage = c10::make_intrusive<at::StorageImpl>(
      at::CTypeToScalarType<real>::to(),
      size,
      state->hipDeviceAllocator,
      true).release();
  return storage;
}

THCStorage* THCStorage_(newWithAllocator)(THCState *state, ptrdiff_t size,
                                          at::Allocator* allocator)
{
  THStorage* storage = c10::make_intrusive<at::StorageImpl>(
      at::CTypeToScalarType<real>::to(),
      size,
      allocator,
      true).release();
  return storage;
}

THCStorage* THCStorage_(newWithSize1)(THCState *state, real data0)
{
  THCStorage *self = THCStorage_(newWithSize)(state, 1);
  THCStorage_(set)(state, self, 0, data0);
  return self;
}

THCStorage* THCStorage_(newWithSize2)(THCState *state, real data0, real data1)
{
  THCStorage *self = THCStorage_(newWithSize)(state, 2);
  THCStorage_(set)(state, self, 0, data0);
  THCStorage_(set)(state, self, 1, data1);
  return self;
}

THCStorage* THCStorage_(newWithSize3)(THCState *state, real data0, real data1, real data2)
{
  THCStorage *self = THCStorage_(newWithSize)(state, 3);
  THCStorage_(set)(state, self, 0, data0);
  THCStorage_(set)(state, self, 1, data1);
  THCStorage_(set)(state, self, 2, data2);
  return self;
}

THCStorage* THCStorage_(newWithSize4)(THCState *state, real data0, real data1, real data2, real data3)
{
  THCStorage *self = THCStorage_(newWithSize)(state, 4);
  THCStorage_(set)(state, self, 0, data0);
  THCStorage_(set)(state, self, 1, data1);
  THCStorage_(set)(state, self, 2, data2);
  THCStorage_(set)(state, self, 3, data3);
  return self;
}

THCStorage* THCStorage_(newWithMapping)(THCState *state, const char *fileName, ptrdiff_t size, int isShared)
{
  THError("not available yet for THCStorage");
  return NULL;
}

THCStorage* THCStorage_(newWithDataAndAllocator)(
    THCState* state,
    at::DataPtr&& data,
    ptrdiff_t size,
    at::Allocator* allocator) {
  THStorage* storage = c10::make_intrusive<at::StorageImpl>(
      at::CTypeToScalarType<real>::to(),
      size,
      std::move(data),
      allocator,
      true).release();
  return storage;
}

void THCStorage_(retain)(THCState *state, THCStorage *self)
{
  THStorage_retain(self);
}

void THCStorage_(free)(THCState *state, THCStorage *self)
{
  THStorage_free(self);
}
#endif
