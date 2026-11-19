/* Dispatch existing ReLU1/Pool1 and new deeper spatial layers through one host scheduler. */
(function(root){
"use strict";
function create(host){
 const first=root.DigitVisionSpatialLayers.create(host),deep=root.DigitVisionAdvancedSpatial.create(host),current=()=>deep.handles(host.S.current.key)?deep:first;
 const facade={handles:key=>first.handles(key)||deep.handles(key),reset:key=>{first.reset(key);deep.reset(key);},pauseTransfer:()=>{first.pauseTransfer();deep.pauseTransfer();},resumeTransfer:()=>current().resumeTransfer(),invalidateGeometry:()=>{first.invalidateGeometry();deep.invalidateGeometry();}};
 for(const name of ["resume","parallelChannel","frame","render","selected","step","restart","modeChanged","openZoom","snapshot"])facade[name]=(...args)=>current()[name](...args);
 Object.defineProperty(facade,"state",{get:()=>current().state});
 return facade;
}
root.DigitVisionUnifiedSpatial={create};
})(typeof window!=="undefined"?window:globalThis);
