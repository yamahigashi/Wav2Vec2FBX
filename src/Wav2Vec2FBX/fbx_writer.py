import os
import sys
import pathlib

import FbxCommon


##############################################################################

if sys.version_info >= (3, 0):
    # For type annotation
    from typing import (  # NOQA: F401 pylint: disable=unused-import
        Optional,
        Dict,
        List,
        Tuple,
        Pattern,
        Callable,
        Any,
        Text,
        Generator,
        Union
    )
##############################################################################


def write(keys, fbx_path):
    # type: (Dict[Text, List], pathlib.Path) -> None

    sdk, scene = FbxCommon.InitializeSdkObjects()
    layer = create_base_anim_structure(scene)

    for node_name, keyframes in keys.items():
        node, curve = add_node(sdk, scene, layer, node_name)

        for frame_sec, val in keyframes:
            curve.KeyModifyBegin()
            set_keyframe(curve, node, frame_sec, val)
            curve.KeyModifyEnd()

    FbxCommon.SaveScene(sdk, scene, fbx_path.as_posix())
    print(f"write fbx to: {fbx_path.as_posix()}")
    sdk.Destroy()


def create_base_anim_structure(scene):
    anim_stack = FbxCommon.FbxAnimStack.Create(scene, "Take 001")
    # anim_stack.LocalStop = FbxCommon.FbxTime(duration_sec)
    anim_layer = FbxCommon.FbxAnimLayer.Create(scene, "Base Layer")
    anim_stack.AddMember(anim_layer)

    return anim_layer


def add_node(sdk, scene, layer, node_name):

    node = FbxCommon.FbxNode.Create(sdk, node_name)
    root = scene.GetRootNode()
    root.AddChild(node)

    prop = FbxCommon.FbxProperty.Create(node, FbxCommon.FbxDouble3DT, "Lcl Translation")
    curve_node = FbxCommon.FbxAnimCurveNode.CreateTypedCurveNode(prop, scene)
    layer.AddMember(curve_node)
    prop.ConnectSrcObject(curve_node)

    x_curve = FbxCommon.FbxAnimCurve.Create(scene, "")
    curve_node.ConnectToChannel(x_curve, 1)

    return node, x_curve


def set_keyframe(curve, node, frame_sec, val):
    time = FbxCommon.FbxTime()
    time.SetSecondDouble(frame_sec)

    key = FbxCommon.FbxAnimCurveKey()
    key.Set(time, val)

    if val == 0.0:
        # set zero key to flat
        key.SetTangentMode(FbxCommon.FbxAnimCurveDef.eTangentUser)
        key.SetTangentWeightMode(FbxCommon.FbxAnimCurveDef.eWeightedNone)
        key.SetDataFloat(FbxCommon.FbxAnimCurveDef.eRightSlope, 0.0)
        key.SetDataFloat(FbxCommon.FbxAnimCurveDef.eNextLeftSlope, 0.0)
        key.SetDataFloat(FbxCommon.FbxAnimCurveDef.eWeights, 0.0)
        key.SetDataFloat(FbxCommon.FbxAnimCurveDef.eRightWeight, 0.333)
        key.SetDataFloat(FbxCommon.FbxAnimCurveDef.eNextLeftWeight, 0.333)

    curve.KeyAdd(time, key)


def test():
    entries = {
        "h": [(0.081015625, 0.0), (0.14177734375, 5.2639923095703125), (0.16203125, 5.293631315231323), (0.22279296875, 0.0)],
        "ow": [(0.1215234375, 0.0), (0.18228515625, 9.779699563980103), (0.243046875, 0.0), (0.5266015625, 0.0), (0.5873632812499999, 9.01805591583252), (0.648125, 0.0)],
        "ao": [(0.1215234375, 0.0), (0.18228515625, 5.943202495574951), (0.243046875, 0.0), (0.5266015625, 0.0), (0.5873632812499999, 4.2488298416137695), (0.648125, 0.0)],
        "o": [(0.1215234375, 0.0), (0.18228515625, 4.948630332946777), (0.243046875, 0.0), (0.30380859375, 0.0), (0.3645703125, 8.939062356948853), (0.42533203124999996, 0.0), (0.5266015625, 0.0), (0.5873632812499999, 4.385859489440918), (0.648125, 0.0)],
        "t": [(0.18228515625, 0.0), (0.243046875, 5.98027229309082), (0.2835546875, 10.0), (0.34431640625, 0.0)],
        "ey": [(0.30380859375, 0.0), (0.3645703125, 4.335820198059082), (0.42533203124999996, 0.0)],
        "y": [(0.34431640625, 0.0), (0.405078125, 5.597909927368164), (0.46583984375, 0.0)],
        "s": [(0.46583984375, 0.0), (0.5266015625, 10.0), (0.5873632812499999, 0.0)],
        "m": [(0.648125, 0.0), (0.70888671875, 5.360048294067383), (0.729140625, 8.003581523895264), (0.74939453125, 10.0), (0.81015625, 0.0)],
    }

    fbx_path = pathlib.Path(os.path.join(os.path.dirname(__file__), "test.fbx"))
    write(entries, fbx_path)


if __name__ == "__main__":
    test()
