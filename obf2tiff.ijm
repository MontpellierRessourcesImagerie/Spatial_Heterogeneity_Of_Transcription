EXT = ".obf";
series = newArray("series_1", "series_2");
names = newArray("NUCLEI", "SPOTS");
dir = getDir("Please select the input folder!");
outDir = dir + "converted/";

setBatchMode(true);
files = getFileList(dir);
images = newArray(0);
for (i = 0; i < files.length; i++) {
	file = files[i];
	if (endsWith(file, EXT)) {
		images = Array.concat(images, file);
	}
}

for (i = 0; i < images.length; i++) {
	showStatus("Converting obf 2 tif");
	showProgress(i+1, images.length);
	image = images[i];
	for (s = 0; s < series.length; s++) {
		currentSeries = series[s];
		path = dir + image;
		run("Bio-Formats Importer", "open=[path] color_mode=Default rois_import=[ROI manager] view=Hyperstack stack_order=Default " + currentSeries);
		resetMinAndMax;
		getVoxelSize(voxelWidth, voxelHeight, voxelDepth, unit);
		if (unit=="microns") {
			setVoxelSize(voxelWidth * 1000, voxelHeight * 1000, voxelDepth * 1000, "nm");
		}
		outfile = replace(image, EXT, "_" + names[s] + ".tif");
		
		if (!File.exists(outDir)) {
			File.makeDirectory(outDir);
		}
		save(outDir + outfile);
		close();
	}
}

setBatchMode("exit and display");