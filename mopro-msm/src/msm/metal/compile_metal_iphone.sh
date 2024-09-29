# every kernel is self contained i.e. is its own crate. Simply `cd` into a kernel's directory and run
# the following, which compiles our shader to an intermediate representation using the metal utility
xcrun -sdk iphoneos metal -c ./mopro-msm/src/msm/metal/shader/all.metal -o ./mopro-msm/src/msm/metal/shader/all.air

# next, compile the .air file to generate a .metallib file - which I believe is LLVM IR (need confirmation)
xcrun -sdk iphoneos metallib ./mopro-msm/src/msm/metal/shader/all.air -o ./mopro-msm/src/msm/metal/shader/msm.metallib

# finally, clean the redundant .air file
rm -f ./mopro-msm/src/msm/metal/shader/all.air