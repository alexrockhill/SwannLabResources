#!/bin/bash
while (( "$#" )); do
  case "$1" in
    -s|--subject)
      SUBJECT=$2
      shift 2
      ;;
    -n|--session)
      SESSION=$2
      shift 2
      ;;
    -r|--run)
      RUN=$2
      shift 2
      ;;
    -t|--task)
      TASK=$2
      shift 2
      ;;
    -b|--behf)
      BEHF=$2
      shift 2
      ;;
    -e|--eegf)
      EEGF=$2
      shift 2
      ;;
    -d|--dir)
      DIR=$2
      shift 2
      ;;
    --) # end argument parsing
      shift
      break
      ;;
    -*|--*=) # unsupported flags
      echo "Error: Unsupported flag $1" >&2
      exit 1
      ;;
  esac
done

my_dir=`pwd`
BASE_NAME="sub-${SUBJECT}"
if [ ! -z $SESSION ]
then
  BASE_NAME="${BASE_NAME}_ses-${SESSION}"
fi
BASE_NAME="${BASE_NAME}_task-${TASK}"
if [ ! -z $RUN ]
then
  BASE_NAME="${BASE_NAME}_run-${RUN}"
fi

echo "Copying electrophysiology data to BIDS directory"
#python save2bids.py $BASE_NAME $EEGF $DIR

if [ ! -z "$TASK" ]
then
	echo "Processing behavior"
  echo "Converting from MATLAB"
	$MATLAB_ROOT/bin/matlab -nodisplay -nosplash -nodesktop -r "cd '$my_dir'; slow_fast_mat2csv('$BEHF'); quit;"
  echo "Copying to BIDS directory"
  if [ -z $SESSION ]
  then
    mkdir -p "$DIR/sub-$SUBJECT/beh"
    cp ${BEHF%.mat}.tsv $DIR/sub-$SUBJECT/beh/${BASE_NAME}_beh.tsv
    python add_beh_sidecar.py "$DIR/sub-$SUBJECT/beh/${BASE_NAME}" "${BEHF%.mat}parameters.tsv" "$DIR"
  else
    mkdir -p "$DIR/sub-$SUBJECT/ses-$SESSION/beh"
    cp ${BEHF%.mat}.tsv $DIR/sub-$SUBJECT/ses-$SESSION/beh/${BASE_NAME}_beh.tsv
    python add_beh_sidecar.py "$DIR/sub-$SUBJECT/ses-$SESSION/beh/${BASE_NAME}" "${BEHF%.mat}parameters.tsv" "$DIR"
  fi
	
fi

echo "Done"