#!/bin/bash

cd '/m100_work/ICT22_ESP_0/vblasone/ERA5/'

for v in 'q' 't' 'u' 'v' 'z' ; do
    files=$(ls ${v}_*.nc)
    for file in $files ; do
        cdo sellonlatbox,6.0,19.25,36.0,47.75 /m100_work/ICT22_ESP_0/vblasone/ERA5/${file} /m100_work/ICT22_ESP_0/vblasone/SLICED/${file}_sliced.nc
    done
done


cd '/m100_work/ICT22_ESP_0/vblasone/SLICED/'

for v in 'q' 't' 'u' 'v' 'z' ; do
    cdo -O -f nc4 -z zip -L -b F32 mergetime ${v}_2001.nc_sliced.nc ${v}_2002.nc_sliced.nc ${v}_2003.nc_sliced.nc ${v}_2004.nc_sliced.nc ${v}_2005.nc_sliced.nc ${v}_2006.nc_sliced.nc ${v}_2007.nc_sliced.nc ${v}_2008.nc_sliced.nc ${v}_2009.nc_sliced.nc ${v}_2010.nc_sliced.nc ${v}_2011.nc_sliced.nc ${v}_2012.nc_sliced.nc ${v}_2013.nc_sliced.nc ${v}_2014.nc_sliced.nc ${v}_2015.nc_sliced.nc ${v}_2016.nc_sliced.nc ${v}_sliced.nc
done
