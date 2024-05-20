// import { app } from '/web/scripts/app.js'
// import { api } from '/web/scripts/api.js'
// // import { ExtendedComfyWidgets,showniftiInput } from "./extended_widgets.js";
// const MultilineSymbol = Symbol();
// const MultilineResizeSymbol = Symbol();


// async function uploadFile(file, updateNode, node, pasted = false) {
// 	const niftiWidget = node.widgets.find((w) => w.name === "nifti_file");
// 	try {
// 		// Wrap file in formdata so it includes filename
// 		const body = new FormData();
// 		body.append("image", file);
// 		if (pasted) {
// 			body.append("subfolder", "nifti");
// 		}
// 		else {
// 			body.append("subfolder", "nifti");
// 		}
	
// 		const resp = await api.fetchApi("/upload/image", {
// 			method: "POST",
// 			body,
// 		});

// 		if (resp.status === 200) {
// 			const data = await resp.json();
// 			// Add the file to the dropdown list and update the widget value
// 			let path = data.name;
			

// 			if (!niftiWidget.options.values.includes(path)) {
// 				niftiWidget.options.values.push(path);
// 			}
			
// 			if (updateNode) {
		
// 				niftiWidget.value = path;
// 				if (data.subfolder) path = data.subfolder + "/" + path;
// 				// showniftiInput(path,node);
				
// 			}
// 		} else {
// 			alert(resp.status + " - " + resp.statusText);
// 		}
// 	} catch (error) {
// 		alert(error);
// 	}
// }




// let uploadWidget = "";
// app.registerExtension({
// 	name: "Comfy.niftiLoad",
// 	async beforeRegisterNodeDef(nodeType, nodeData, app) {

// 		const onAdded = nodeType.prototype.onAdded;
// 		if (nodeData.name === "LoadNifti") {
// 		nodeType.prototype.onAdded = function () {
// 			onAdded?.apply(this, arguments);
// 			// const temp_web_url = this.widgets.find((w) => w.name === "local_url");
// 			// const autoplay_value = this.widgets.find((w) => w.name === "autoplay");
		
			
// 			let uploadWidget;
// 			const fileInput = document.createElement("input");
// 			Object.assign(fileInput, {
// 				type: "file",
// 				accept: "nifti/nii,nifti/nii.gz",
// 				style: "display: none",
// 				onchange: async () => {
// 					if (fileInput.files.length) {
// 						// console.log("fileInput.files[0]",fileInput.files[0])
// 						await uploadFile(fileInput.files[0], true, this);
// 					}
// 				},
// 			});
// 			document.body.append(fileInput);
// 			// Create the button widget for selecting the files
// 			uploadWidget = this.addWidget("button", "choose file to upload", "image", () => {
// 				fileInput.click();
// 			},{
// 				cursor: "grab",
// 			},);
// 			uploadWidget.serialize = false;


// 		// setTimeout(() => {
// 		// 	ExtendedComfyWidgets["nifti"](this, "niftiWidget", ["STRING"], temp_web_url.value, app,"input", autoplay_value.value);
		
// 		// }, 100); 
		
		
// 		}
// 			nodeType.prototype.onDragOver = function (e) {
// 				if (e.dataTransfer && e.dataTransfer.items) {
// 					const image = [...e.dataTransfer.items].find((f) => f.kind === "file");
// 					return !!image;
// 				}
	
// 				return false;
// 			};
	
			
// 			nodeType.prototype.onDragDrop = function (e) {
// 				console.log("onDragDrop called");
// 				let handled = false;
// 				for (const file of e.dataTransfer.files) {
// 					if (file.type.startsWith("nifti/nii", "nifti/nii.gz")) {
						
// 						const filePath = file.path || (file.webkitRelativePath || '').split('/').slice(1).join('/'); 


// 						uploadFile(file, !handled,this ); // Dont await these, any order is fine, only update on first one

// 						handled = true;
// 					}
// 				}
	
// 				return handled;
// 			};
	
// 			nodeType.prototype.pasteFile = function(file) {
// 				if (file.type.startsWith("nifti/nii", "nifti/nii.gz")) {

// 					uploadFile(file, true, is_pasted);

// 					return true;
// 				}
// 				return false;
// 			}


// 		};
		
// 	},
// });


import { app } from "/scripts/app.js";
import { api } from "/scripts/api.js";
//import { ExtendedComfyWidgets,showniftiInput } from "./extended_widgets.js";
const MultilineSymbol = Symbol();
const MultilineResizeSymbol = Symbol();

async function uploadFile(file, updateNode, node) {
	const niftiWidget = node.widgets.find((w) => w.name === "nifti_file");

	try {
		// Wrap file in formdata so it includes filename
		const body = new FormData();
		body.append("image", file);
		body.append("subfolder", "nifti");
		console.log('body',body);
	
		const resp = await api.fetchApi("/upload/image", {
			method: "POST",
			body,
		});

		if (resp.status === 200) {
			const data = await resp.json();
			// Add the file to the dropdown list and update the widget value
			let path = data.name;
			console.log('path',path);
			console.log('niftiWidget.options.values', niftiWidget.options.values);

		
			if (!niftiWidget.options.values.includes(path)) {
				niftiWidget.options.values.push(path);
			}
			
			if (updateNode) {
		
				niftiWidget.value = path;
				if (data.subfolder) path = data.subfolder + "/" + path;
				// showniftiInput(path,node);
				
			}
		} else {
			alert(resp.status + " - " + resp.statusText);
		}
	} catch (error) {
		alert(error);
	}
}



let uploadWidget = "";
app.registerExtension({
	name: "Comfy.niftiLoad",
	async beforeRegisterNodeDef(nodeType, nodeData, app) {

		const onAdded = nodeType.prototype.onAdded;
		if (nodeData.name === "LoadNifti") {

		nodeType.prototype.onAdded = function () {
			onAdded?.apply(this, arguments);
			// const temp_web_url = this.widgets.find((w) => w.name === "local_url");
			// const autoplay_value = this.widgets.find((w) => w.name === "autoplay");
		
			
			let uploadWidget;
			const fileInput = document.createElement("input");
			Object.assign(fileInput, {
				type: "file",
				accept: "nifti/nii,nifti/nii.gz",
				style: "display: none",
				onchange: async () => {
					if (fileInput.files.length) {
						
						
						let file=fileInput.files[0];
			
						console.log("fileInput.files[0]",file);
						await uploadFile(file, true, this);
					}
				},
			});
			document.body.append(fileInput);
			// Create the button widget for selecting the files
			uploadWidget = this.addWidget("button", "choose file to upload", "image", () => {
				fileInput.click();
			},{
				cursor: "grab",
			},);
			uploadWidget.serialize = false;


		//setTimeout(() => {
		//	ExtendedComfyWidgets["nifti"](this, "niftiWidget", ["STRING"], temp_web_url.value, app,"input", autoplay_value.value);
		
		//}, 100); 
		
		
		}
			nodeType.prototype.onDragOver = function (e) {
				if (e.dataTransfer && e.dataTransfer.items) {
					const image = [...e.dataTransfer.items].find((f) => f.kind === "file");
					return !!image;
				}
	
				return false;
			};
	
			nodeType.prototype.onDragDrop = function (e) {
				console.log("onDragDrop called");
				let handled = false;
				for (const file of e.dataTransfer.files) {
					if (file.type.startsWith("nifti/nii")) {
						
						const filePath = file.path || (file.webkitRelativePath || '').split('/').slice(1).join('/'); 


						uploadFile(file, !handled,this ); // Dont await these, any order is fine, only update on first one

						handled = true;
					}
				}
	
				return handled;
			};
	
			nodeType.prototype.pasteFile = function(file) {
				if (file.type.startsWith("nifti/nii")) {

					// uploadFile(file, true, is_pasted);

					return true;
				}
				return false;
			}


		};
		
	},
});
